"""Observability: JSON logs, request-id middleware, Prometheus metrics, typed errors."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CollectorRegistry, Counter, Histogram
from sqlalchemy.exc import OperationalError
from starlette.exceptions import HTTPException as StarletteHTTPException

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
LOG = logging.getLogger("prahari.api")
BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
STATUS_TYPES = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "validation_error",
    502: "action_failed",
    503: "model_unavailable",
}


class ApiError(Exception):
    """Raised by handlers; rendered as ``{"error": {type, message, request_id}}``."""

    def __init__(self, status: int, message: str, type_: str | None = None):
        super().__init__(message)
        self.status, self.message = status, message
        self.type = type_ or STATUS_TYPES.get(status, "error")


class JsonFormatter(logging.Formatter):
    """One JSON object per log line, with the current request id."""

    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        out.update(getattr(record, "extra_fields", {}))
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging() -> None:
    """Install the JSON formatter on the ``prahari`` logger once."""
    log = logging.getLogger("prahari")
    if not any(isinstance(h.formatter, JsonFormatter) for h in log.handlers):
        h = logging.StreamHandler()
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False


class Metrics:
    """Per-app Prometheus registry (a registry per app keeps tests independent)."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.latency = Histogram(
            "prahari_request_duration_seconds",
            "API latency by endpoint",
            ["method", "path"],
            buckets=BUCKETS,
            registry=self.registry,
        )
        self.requests = Counter(
            "prahari_requests_total",
            "API requests",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.incidents = Counter(
            "prahari_incidents_detected_total", "Incidents opened", registry=self.registry
        )
        self.detector_runs = Counter(
            "prahari_detector_runs_total", "Detector cycles", ["outcome"], registry=self.registry
        )


def _error(status: int, type_: str, message: str, fields=None) -> JSONResponse:
    body = {"type": type_, "message": message, "request_id": request_id_var.get()}
    if fields:
        body["fields"] = fields
    return JSONResponse({"error": body}, status_code=status)


def install(app: FastAPI, metrics: Metrics) -> None:
    """Attach request-id/latency middleware and typed error handlers."""

    @app.middleware("http")
    async def _mw(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = rid
            return response
        finally:
            dt = time.perf_counter() - t0
            route = request.scope.get("route")
            path = getattr(route, "path", None) or "unmatched"
            metrics.latency.labels(request.method, path).observe(dt)
            metrics.requests.labels(request.method, path, str(status)).inc()
            LOG.info(
                "request",
                extra={
                    "extra_fields": {
                        "method": request.method,
                        "path": request.url.path,
                        "status": status,
                        "ms": round(dt * 1000, 2),
                    }
                },
            )
            request_id_var.reset(token)

    @app.exception_handler(ApiError)
    async def _api(_: Request, e: ApiError):
        return _error(e.status, e.type, e.message)

    @app.exception_handler(RequestValidationError)
    async def _val(_: Request, e: RequestValidationError):
        fields = [
            {"loc": [str(x) for x in err["loc"]], "msg": err["msg"], "type": err["type"]}
            for err in e.errors()
        ]
        return _error(422, "validation_error", "request validation failed", fields)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, e: StarletteHTTPException):
        return _error(e.status_code, STATUS_TYPES.get(e.status_code, "error"), str(e.detail))

    @app.exception_handler(OperationalError)
    async def _db(_: Request, e: OperationalError):
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            r = _error(503, "database_busy", "database is busy; retry shortly")
            r.headers["Retry-After"] = "1"
            return r
        LOG.exception("database error")
        return _error(500, "internal_error", "database error")

    @app.exception_handler(Exception)
    async def _any(_: Request, e: Exception):
        LOG.exception("unhandled error")
        return _error(500, "internal_error", f"{type(e).__name__}: {e}")
