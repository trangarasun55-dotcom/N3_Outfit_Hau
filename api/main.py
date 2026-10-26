"""FastAPI inference service: load pipeline and catalog once at module import."""

from __future__ import annotations

import os
import sqlite3
import csv
import io
import logging
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import UUID

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import (
    Body,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Path as ApiPath,
    Query,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
LOGGER = logging.getLogger(__name__)

try:
    from . import storage
    from .llm_explain import add_explanations
    from .auth import (
        COOKIE_NAME,
        get_current_user,
        hash_password,
        issue_session,
        normalize_email,
        require_admin,
        seed_admin_from_environment,
        verify_password,
    )
    from .auth_schemas import (
        LoginRequest,
        PasswordChangeRequest,
        ProfilePatch,
        RegisterRequest,
    )
    from .config import (
        CATALOG_PATH,
        COLOR_CHOICES,
        CONFIDENCE_HIGH_STD,
        CONFIDENCE_MEDIUM_STD,
        MODEL_METADATA,
        MODEL_PATH,
        MODEL_TEMP_RANGE,
    )
    from .recommender import confidence_label, recommend as build_recommendations
    from .schemas import (
        FeedbackRequest,
        FinalizationRequest,
        OutfitRatingRequest,
        PredictRequest,
        RecommendRequest,
        ViewEventRequest,
    )
    from .storage import (
        add_feedback,
        create_user,
        get_user_by_email,
        save_recommendation,
        update_last_login,
        update_password_hash,
        update_user_profile,
    )
    from .weather import get_weather, geocode
except ImportError:  # chạy `uvicorn main:app` từ thư mục api
    import storage
    from llm_explain import add_explanations
    from auth import (
        COOKIE_NAME,
        get_current_user,
        hash_password,
        issue_session,
        normalize_email,
        require_admin,
        seed_admin_from_environment,
        verify_password,
    )
    from auth_schemas import (
        LoginRequest,
        PasswordChangeRequest,
        ProfilePatch,
        RegisterRequest,
    )
    from config import (
        CATALOG_PATH,
        COLOR_CHOICES,
        CONFIDENCE_HIGH_STD,
        CONFIDENCE_MEDIUM_STD,
        MODEL_METADATA,
        MODEL_PATH,
        MODEL_TEMP_RANGE,
    )
    from recommender import confidence_label, recommend as build_recommendations
    from schemas import (
        FeedbackRequest,
        FinalizationRequest,
        OutfitRatingRequest,
        PredictRequest,
        RecommendRequest,
        ViewEventRequest,
    )
    from storage import (
        add_feedback,
        create_user,
        get_user_by_email,
        save_recommendation,
        update_last_login,
        update_password_hash,
        update_user_profile,
    )
    from weather import get_weather, geocode


try:
    MODEL_ARTIFACT: dict[str, Any] = joblib.load(MODEL_PATH)
    CATALOG = pd.read_csv(
        CATALOG_PATH,
        encoding="utf-8-sig",
        dtype={"ID_SP": "string"},
    )
except Exception as exc:
    raise RuntimeError(
        "Không nạp được artifact Bước 1; hãy chạy `python model/train_export.py` và kiểm tra file model/metadata/data."
    ) from exc

if len(CATALOG) == 0:
    raise RuntimeError("Danh mục data/mon_do.csv rỗng; hãy chạy lại Bước 1.")

@asynccontextmanager
async def lifespan(_: FastAPI):
    storage.initialize_storage()
    seed_admin_from_environment()
    yield


app = FastAPI(
    title="OutfitAI REST API",
    description="API OutfitAI: suy luận, giải thích tùy chọn, thời tiết và lưu lịch sử/phản hồi SQLite.",
    version=MODEL_METADATA["version"],
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(
            "CORS_ORIGINS", "http://127.0.0.1:5000,http://localhost:5000"
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["*"],
)


def _set_session_cookie(response: Response, user_id: str) -> None:
    token, max_age = issue_session(user_id)
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=max_age,
        httponly=True,
        secure=os.environ.get("COOKIE_SECURE", "0").strip().casefold()
        in {"1", "true", "yes"},
        samesite="lax",
        path="/",
    )


@app.post("/auth/register", status_code=201)
def register(request: RegisterRequest, response: Response) -> dict[str, Any]:
    email = normalize_email(request.email)
    try:
        user = create_user(
            email=email,
            display_name=request.display_name or email.split("@", 1)[0],
            password_hash=hash_password(request.password),
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="Email này đã được đăng ký.",
        ) from exc
    _set_session_cookie(response, user["id"])
    return {"user": user}


@app.post("/auth/login")
def login(request: LoginRequest, response: Response) -> dict[str, Any]:
    user = get_user_by_email(normalize_email(request.email))
    if (
        user is None
        or not user["is_active"]
        or not verify_password(request.password, user["password_hash"])
    ):
        raise HTTPException(status_code=401, detail="Email hoặc mật khẩu không đúng.")
    update_last_login(user["id"])
    refreshed_user = storage.get_user_by_id(user["id"])
    if refreshed_user is None:
        raise RuntimeError("Không tìm thấy tài khoản sau khi cập nhật đăng nhập.")
    _set_session_cookie(response, user["id"])
    return {"user": refreshed_user}


@app.post("/auth/logout")
def logout(response: Response) -> dict[str, str]:
    response.delete_cookie(
        COOKIE_NAME,
        path="/",
        httponly=True,
        secure=os.environ.get("COOKIE_SECURE", "0").strip().casefold()
        in {"1", "true", "yes"},
        samesite="lax",
    )
    return {"status": "logged_out"}


@app.get("/auth/me")
def auth_me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"user": user}


@app.get("/profile")
def get_profile(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"profile": user}


@app.patch("/profile")
def patch_profile(
    request: ProfilePatch,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    updates = request.model_dump(exclude_unset=True)
    if "display_name" in updates and updates["display_name"] is None:
        raise HTTPException(status_code=422, detail="Tên hiển thị không được để trống.")
    profile = update_user_profile(user["id"], updates)
    if profile is None:
        raise HTTPException(status_code=401, detail="Tài khoản không còn hoạt động.")
    return {"profile": profile}


@app.post("/profile/password")
def change_password(
    request: PasswordChangeRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, str]:
    stored_user = get_user_by_email(user["email"])
    if stored_user is None or not verify_password(
        request.current_password, stored_user["password_hash"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Mật khẩu hiện tại không đúng.",
        )
    update_password_hash(user["id"], hash_password(request.new_password))
    return {"status": "password_updated"}


@app.get("/admin/users")
def admin_users(
    _: dict[str, Any] = Depends(require_admin),
    limit: int = Query(default=100, ge=1, le=500),
    page: int = Query(default=1, ge=1),
) -> dict[str, Any]:
    return {
        "items": storage.admin_users(limit=limit, offset=(page - 1) * limit),
        "page": page,
        "page_size": limit,
    }


@app.get("/admin/stats")
def admin_statistics(
    _: dict[str, Any] = Depends(require_admin),
) -> dict[str, Any]:
    return {"stats": storage.admin_stats()}


@app.get("/admin/outfits")
def admin_outfit_statistics(
    _: dict[str, Any] = Depends(require_admin),
    limit: int = Query(default=100, ge=1, le=500),
    page: int = Query(default=1, ge=1),
) -> dict[str, Any]:
    return {
        "items": storage.admin_outfits(limit=limit, offset=(page - 1) * limit),
        "page": page,
        "page_size": limit,
    }


@app.get("/admin/export.csv")
def admin_export_csv(
    _: dict[str, Any] = Depends(require_admin),
) -> StreamingResponse:
    rows = storage.admin_outfits(limit=100_000)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        [
            "outfit_id",
            "composition_key",
            "name",
            "recommendation_count",
            "view_count",
            "view_seconds",
            "finalize_count",
            "likes",
            "dislikes",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row["outfit_id"],
                row["composition_key"],
                row["name"],
                row["recommendation_count"],
                row["view_count"],
                row["view_seconds"],
                row["finalize_count"],
                row["likes"] or 0,
                row["dislikes"] or 0,
            ]
        )
    return StreamingResponse(
        iter(["\ufeff" + output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="outfitai-admin.csv"'},
    )


def _predict_features(feature_values: dict[str, object]) -> dict[str, Any]:
    feature_frame = pd.DataFrame(
        [feature_values], columns=MODEL_METADATA["features"]
    )
    rf_pipe = MODEL_ARTIFACT["rf_pipe"]
    transformed = rf_pipe.named_steps["preprocess"].transform(feature_frame)
    rf_model = rf_pipe.named_steps["model"]
    tree_scores = np.asarray(
        [tree.predict(transformed)[0] for tree in rf_model.estimators_], dtype=float
    )
    rf_score = float(tree_scores.mean())
    std = float(tree_scores.std(ddof=0))
    if MODEL_ARTIFACT.get("selected_model") == "Lai":
        knn_score = float(MODEL_ARTIFACT["knn_pipe"].predict(feature_frame)[0])
        weight = float(MODEL_ARTIFACT["w"])
        score = weight * rf_score + (1 - weight) * knn_score
    else:
        score = rf_score
    score = float(np.clip(score, 0, 10))
    return {
        "score": score,
        "score_low": float(max(0, score - std)),
        "score_high": float(min(10, score + std)),
        "std_between_trees": std,
        "confidence_level": confidence_label(std),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_version": str(MODEL_METADATA["version"]),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    # The public discovery endpoint serves UI choices only. Keep held-out metrics,
    # split membership, and training diagnostics in the local artifact/report.
    public_keys = {
        "model_name", "version", "features", "target", "categories",
        "feature_ranges",
    }
    info = {key: deepcopy(MODEL_METADATA[key]) for key in public_keys if key in MODEL_METADATA}
    info["categories"]["Màu giao diện"] = list(COLOR_CHOICES)
    info["confidence_thresholds"] = {
        "high_std_max": CONFIDENCE_HIGH_STD,
        "medium_std_max": CONFIDENCE_MEDIUM_STD,
    }
    return info


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    response = _predict_features(request.as_features())
    response["model_version"] = MODEL_METADATA["version"]
    if not (
        MODEL_TEMP_RANGE["min"]
        <= request.temperature
        <= MODEL_TEMP_RANGE["max"]
    ):
        response["confidence_level"] = "Thấp"
        response["warnings"] = [
            f"Nhiệt độ ngoài khoảng huấn luyện {MODEL_TEMP_RANGE['min']:g}–{MODEL_TEMP_RANGE['max']:g}°C; kết quả ngoại suy."
        ]
    else:
        response["warnings"] = []
    return response


def _authenticated_user_id(
    user: dict[str, Any], supplied_user_id: str | None
) -> UUID:
    authenticated_id = UUID(user["id"])
    if supplied_user_id is None:
        return authenticated_id
    try:
        requested_id = UUID(supplied_user_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="X-User-Id phải là UUID hợp lệ."
        ) from exc
    if requested_id != authenticated_id:
        raise HTTPException(
            status_code=403,
            detail="X-User-Id không khớp với tài khoản đã đăng nhập.",
        )
    return authenticated_id


@app.post("/recommend/preview")
def recommend_preview(
    request: RecommendRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Preview recommendations for live UI updates without polluting user history."""
    preferred_style = user.get("preferred_style")
    if not request.styles and preferred_style in MODEL_METADATA["categories"]["Phong cách"]:
        request = request.model_copy(update={"styles": [preferred_style]})
    result = build_recommendations(CATALOG, MODEL_ARTIFACT, request)
    result["preview"] = True
    return result


@app.post("/recommend")
def recommend(
    request: RecommendRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    user_id = _authenticated_user_id(user, x_user_id)
    preferred_style = user.get("preferred_style")
    if not request.styles and preferred_style in MODEL_METADATA["categories"]["Phong cách"]:
        request = request.model_copy(update={"styles": [preferred_style]})
    result = build_recommendations(CATALOG, MODEL_ARTIFACT, request)
    try:
        # Lý do theo luật đi cùng top 3; Gemini chạy sau trên endpoint riêng.
        result = add_explanations(result, request.model_dump(), use_gemini=False)
    except Exception as exc:  # lời giải thích AI là tùy chọn, không làm hỏng kết quả DSS
        LOGGER.warning("Gemini explanation integration failed (%s)", type(exc).__name__)
        for outfit in result.get("outfits", []):
            outfit["explanation_llm"] = str(
                outfit.get("explanation") or outfit.get("recommendation") or "Chưa có lời giải thích."
            )
            outfit["explanation_structured"] = None
            outfit["explanation_source"] = "rules"
            outfit["comment_source"] = "rules"
            outfit["explanation_notice"] = "Gemini không khả dụng; hiện đang dùng giải thích theo quy tắc."
    save_recommendation(user_id, request.model_dump(), result)
    return result


@app.post("/recommendations/{request_id}/explanations")
def recommendation_explanations(
    request_id: UUID,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Upgrade rule-based explanations for one saved top-3 recommendation."""
    user_id = UUID(user["id"])
    is_admin = user["role"] == "admin"
    try:
        saved = storage.get_recommendation_for_explanation(
            user_id, request_id, is_admin=is_admin
        )
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="Không đọc được dữ liệu gợi ý đã lưu.") from exc
    if saved is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy gợi ý này.")
    request_data, result = saved
    outfits = result.get("outfits", [])
    if not isinstance(outfits, list) or not outfits:
        raise HTTPException(status_code=422, detail="Gợi ý đã lưu không có bộ trang phục.")
    if all(outfit.get("explanation_source") == "gemini" for outfit in outfits):
        return {
            "request_id": str(request_id),
            "outfits": [
                {
                    "id": outfit.get("id"),
                    **{
                        key: outfit.get(key)
                        for key in (
                            "explanation_llm", "explanation_structured", "explanation_source",
                            "comment_source", "explanation_notice",
                        )
                    },
                }
                for outfit in outfits
            ],
        }
    try:
        result = add_explanations(result, request_data)
    except Exception as exc:
        LOGGER.warning("Gemini explanation endpoint failed (%s)", type(exc).__name__)
        result = add_explanations(result, request_data, use_gemini=False)
    try:
        saved_ok = storage.update_recommendation_explanations(
            user_id,
            request_id,
            result.get("outfits", []),
            is_admin=is_admin,
        )
    except Exception as exc:
        LOGGER.warning("Could not persist recommendation explanations (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="Chưa lưu được phần giải thích; trang vẫn giữ lời giải thích theo quy tắc.",
        ) from exc
    if not saved_ok:
        raise HTTPException(status_code=404, detail="Không tìm thấy gợi ý này.")
    return {
        "request_id": str(request_id),
        "outfits": [
            {
                "id": outfit.get("id"),
                **{
                    key: outfit.get(key)
                    for key in (
                        "explanation_llm", "explanation_structured", "explanation_source",
                        "comment_source", "explanation_notice",
                    )
                },
            }
            for outfit in result.get("outfits", [])
        ],
    }


@app.get("/history")
def recommendation_history(
    user: dict[str, Any] = Depends(get_current_user),
    history_type: str = Query(
        default="viewed", alias="type", pattern="^(viewed|finalized|feedback)$"
    ),
    range_name: str = Query(
        default="all", alias="range", pattern="^(day|week|month|all|from-to)$"
    ),
    from_date: str | None = Query(default=None, alias="from", max_length=10),
    to_date: str | None = Query(default=None, alias="to", max_length=10),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    try:
        return storage.get_history_page(
            user_id=UUID(user["id"]),
            is_admin=user["role"] == "admin",
            history_type=history_type,
            range_name=range_name,
            from_date=from_date,
            to_date=to_date,
            page=page,
            page_size=page_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/events/view")
def outfit_view_event(
    request: ViewEventRequest,
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    user_id = UUID(user["id"])
    is_admin = user["role"] == "admin"
    if request.action == "start":
        assert request.request_item_id is not None
        event = storage.start_outfit_view(
            user_id, request.request_item_id, is_admin=is_admin
        )
    else:
        assert request.view_id is not None
        event = storage.update_outfit_view(
            user_id, request.view_id, request.action, is_admin=is_admin
        )
    if event is None:
        raise HTTPException(
            status_code=404, detail="Không tìm thấy bộ trang phục hoặc lượt xem."
        )
    return event


@app.post("/outfits/{outfit_id}/finalize")
def finalize_outfit(
    outfit_id: int = ApiPath(ge=1),
    request: FinalizationRequest = Body(default=FinalizationRequest()),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    finalized = storage.finalize_outfit(
        UUID(user["id"]),
        outfit_id,
        request.note,
        is_admin=user["role"] == "admin",
    )
    if finalized is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy bộ trang phục.")
    return {"status": "finalized", "finalization": finalized}


@app.delete("/outfits/{outfit_id}/finalize")
def cancel_finalize_outfit(
    outfit_id: int = ApiPath(ge=1),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, str]:
    if not storage.cancel_finalization(
        UUID(user["id"]), outfit_id, is_admin=user["role"] == "admin"
    ):
        raise HTTPException(status_code=404, detail="Không tìm thấy bộ trang phục.")
    return {"status": "unfinalized"}


@app.post("/outfits/{outfit_id}/feedback")
def rate_outfit(
    request: OutfitRatingRequest,
    outfit_id: int = ApiPath(ge=1),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    rating = storage.rate_outfit(
        UUID(user["id"]),
        outfit_id,
        request.value,
        request.reason,
        is_admin=user["role"] == "admin",
    )
    if rating is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy bộ trang phục.")
    return {"status": "saved", "feedback": rating}


@app.post("/feedback")
def submit_feedback(
    request: FeedbackRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
    user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, str]:
    user_id = _authenticated_user_id(user, x_user_id)
    feedback_id = add_feedback(
        user_id=user_id,
        request_id=request.request_id,
        outfit_id=request.outfit_id,
        action=request.action,
        note=request.note,
    )
    if feedback_id == "not_found":
        raise HTTPException(status_code=404, detail="Không tìm thấy lượt gợi ý của người dùng này.")
    if feedback_id == "outfit_not_found":
        raise HTTPException(status_code=404, detail="Không tìm thấy bộ trang phục trong lượt gợi ý.")
    return {"status": "saved", "feedback_id": feedback_id}


@app.get("/weather")
def weather(
    lat: float | None = Query(default=None, ge=-90, le=90),
    lon: float | None = Query(default=None, ge=-180, le=180),
    city: str | None = Query(default=None, max_length=80),
) -> dict[str, Any]:
    """Current weather and today's forecast, fetched server-side from Open-Meteo."""
    return get_weather(lat=lat, lon=lon, city=city)


@app.get("/geocode")
def geocode_city(q: str = Query(min_length=1, max_length=80)) -> dict[str, Any]:
    return {"results": geocode(q)}
