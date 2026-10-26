"""Flask UI client for the OutfitAI REST API."""

import io
import math
import os
import re
import secrets
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import requests
from dotenv import load_dotenv
from flask import Flask, Response, flash, jsonify, redirect, render_template, request, session, url_for
from markupsafe import Markup, escape
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def load_secret_key():
    """SECRET_KEY từ .env; nếu trống thì tạo một lần và lưu lại để phiên không mất khi khởi động lại."""
    configured = os.environ.get("SECRET_KEY", "").strip()
    if configured:
        return configured
    key_file = ROOT / "data" / ".secret_key"
    try:
        if key_file.is_file():
            stored = key_file.read_text(encoding="utf-8").strip()
            if stored:
                return stored
        key_file.parent.mkdir(parents=True, exist_ok=True)
        generated = secrets.token_hex(32)
        key_file.write_text(generated, encoding="utf-8")
        return generated
    except OSError:
        return secrets.token_hex(32)


app = Flask(__name__)
app.secret_key = load_secret_key()
app.config["MAX_CONTENT_LENGTH"] = 6 * 1024 * 1024
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 3600
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


def outfit_emphasis(value):
    """Escape model text first, then allow only the requested **bold** markers."""
    escaped_text = str(escape("" if value is None else str(value)))
    safe_html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped_text)
    return Markup(safe_html)


app.jinja_env.filters["outfit_emphasis"] = outfit_emphasis

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = (5, 15)
RECOMMEND_TIMEOUT = (5, 60)
EXPLANATION_TIMEOUT = (3, 22)
PUBLIC_ENDPOINTS = {"login", "register", "static", "healthz", "favicon"}
OUTFIT_RESULTS = OrderedDict()
OUTFIT_RESULTS_MAX = 128

# UI labels and descriptions; accepted values are always checked against API metadata.
ACTIVITY_UI = {
    "Đi học": {"icon": "📚", "description": "Buổi học trên lớp, studio, seminar", "chip": "Lý thuyết"},
    "Xưởng/Công trình": {"icon": "⌖", "description": "Làm mô hình, workshop xưởng, tiếp xúc vật liệu, công trình", "chip": "Thực hành"},
    "Vẽ ngoại cảnh": {"icon": "⌖", "description": "Ký họa, khảo sát ngoài trời, hiện trường", "chip": "Ngoài trời"},
    "Thuyết trình": {"icon": "▣", "description": "Bảo vệ đồ án, báo cáo, sự kiện", "chip": "Báo cáo"},
}
FALLBACK_ACTIVITIES = list(ACTIVITY_UI)
FALLBACK_STYLES = ["Casual", "Smart Casual", "Sporty", "Streetwear", "Minimalist", "Artsy"]
FALLBACK_COLORS = [
    "Bạc", "Be", "Be sữa", "Cam", "Đen", "Đen trắng", "Đỏ", "Đỏ đen", "Đỏ đô",
    "Hồng", "Hồng pastel", "Kem", "Khaki", "Nâu", "Nâu caramel", "Nâu đậm",
    "Nhiều màu", "Tím", "Tím lavender", "Trắng", "Trong suốt", "Vàng",
    "Vàng mustard", "Xám", "Xám tro", "Xanh baby", "Xanh denim đậm",
    "Xanh denim nhạt", "Xanh dương", "Xanh lá", "Xanh mint", "Xanh navy",
    "Xanh olive", "Xanh rêu", "Xanh trắng",
]
FALLBACK_COLOR_GROUPS = ["Đen", "Trắng", "Xám", "Be", "Nâu đất", "Navy"]


def api_get(path, *, params=None, headers=None, timeout=REQUEST_TIMEOUT):
    """GET tới API; thử lại một lần khi mất kết nối thoáng qua (API vừa khởi động lại)."""
    for attempt in (1, 2):
        try:
            response = requests.get(f"{API_URL}{path}", params=params, headers=headers, timeout=timeout)
            break
        except requests.ConnectionError:
            if attempt == 2:
                raise
            time.sleep(0.6)
    response.raise_for_status()
    return response.json()


def cache_outfit_result(result_id, result):
    OUTFIT_RESULTS[result_id] = result
    OUTFIT_RESULTS.move_to_end(result_id)
    while len(OUTFIT_RESULTS) > OUTFIT_RESULTS_MAX:
        OUTFIT_RESULTS.popitem(last=False)


def get_outfit_result(result_id):
    result = OUTFIT_RESULTS.get(result_id)
    if result is not None:
        OUTFIT_RESULTS.move_to_end(result_id)
    return result


def api_user_headers():
    headers = {}
    user_id = session.get("api_user", {}).get("id")
    if user_id:
        headers["X-User-Id"] = user_id
    token = session.get("api_auth_cookie")
    if token:
        headers["Cookie"] = f"outfit_session={token}"
    return headers


def api_error_status(exc, default=503):
    response = getattr(exc, "response", None)
    return response.status_code if response is not None else default


def _auth_request(endpoint, payload):
    response = requests.post(
        f"{API_URL}{endpoint}",
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    cookie = getattr(response, "cookies", {}).get("outfit_session")
    if not cookie:
        raise ValueError("API xác thực chưa cấp phiên đăng nhập.")
    session["api_auth_cookie"] = cookie
    session["api_user"] = data.get("user", {})
    session["user_id"] = session["api_user"].get("id", "")
    session["profile"] = {
        "name": session["api_user"].get("display_name", "Người dùng"),
        "gender": session["api_user"].get("gender") or "",
    }
    return data


def get_model_info():
    """Return API metadata; use UI fallback values only when API is unavailable."""
    try:
        data = api_get("/model-info")
        categories = data.get("categories", {})
        valid_activities = categories.get("Hoàn cảnh") or FALLBACK_ACTIVITIES
        activities = [value for value in FALLBACK_ACTIVITIES if value in valid_activities]
        styles = categories.get("Phong cách") or FALLBACK_STYLES
        colors = categories.get("Màu sắc") or categories.get("Màu giao diện") or FALLBACK_COLORS
        accepted_colors = list(dict.fromkeys(
            [*colors, *(categories.get("Màu giao diện") or FALLBACK_COLOR_GROUPS)]
        ))
        gender = categories.get("Giới tính") or ["Nam", "Nữ", "Unisex"]
        return {"activities": activities, "styles": styles, "colors": colors,
                "accepted_colors": accepted_colors,
                "genders": [value for value in gender if value in {"Nam", "Nữ", "Unisex"}],
                "available": True, "error": None, "metadata": data}
    except (requests.RequestException, ValueError, TypeError):
        return {"activities": FALLBACK_ACTIVITIES, "styles": FALLBACK_STYLES,
                "colors": FALLBACK_COLORS,
                "accepted_colors": list(dict.fromkeys([*FALLBACK_COLORS, *FALLBACK_COLOR_GROUPS])),
                "genders": ["Nam", "Nữ", "Unisex"], "available": False,
                "error": "Không kết nối được API gợi ý. Dữ liệu lựa chọn đang hiển thị theo bộ giá trị OutfitAI; hãy thử lại sau.",
                "metadata": {}}


@app.context_processor
def inject_profile():
    api_user = session.get("api_user", {})
    profile = {
        "name": api_user.get("display_name") or session.get("profile", {}).get("name", "Người dùng"),
        "gender": api_user.get("gender") or session.get("profile", {}).get("gender", ""),
    }
    return {
        "shell_profile": profile,
        "api_user": api_user or None,
        "is_admin": api_user.get("role") == "admin",
    }


def is_logged_in():
    return bool(session.get("api_auth_cookie") and session.get("api_user"))


def safe_next_url(target):
    """Chỉ chấp nhận đường dẫn nội bộ để tránh chuyển hướng ra ngoài."""
    if not target:
        return None
    parts = urlsplit(target)
    if parts.scheme or parts.netloc or not target.startswith("/") or target.startswith("//"):
        return None
    if parts.path in {"/login", "/register", "/logout"}:
        return None
    return target


@app.before_request
def require_login():
    """Mọi trang ngoài đăng nhập/đăng ký đều cần phiên đăng nhập."""
    if request.endpoint in PUBLIC_ENDPOINTS:
        return None
    if is_logged_in():
        return None
    if request.path.startswith("/ui/"):
        return jsonify({"message": "Phiên đăng nhập đã hết hạn. Hãy đăng nhập lại."}), 401
    session.clear()
    next_url = None
    if request.method == "GET":
        next_url = safe_next_url(request.full_path.rstrip("?"))
    if next_url and next_url != "/":
        return redirect(url_for("login", next=next_url))
    return redirect(url_for("login"))


@app.after_request
def disable_page_caching(response):
    """Không cho trình duyệt lưu trang đã đăng nhập (nút Back sau khi đăng xuất không lộ nội dung)."""
    if request.endpoint != "static":
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.get("/favicon.ico")
def favicon():
    return "", 204


def friendly_register_error(exc):
    """Chuyển lỗi 422 của API thành thông báo tiếng Việt cụ thể."""
    try:
        detail = exc.response.json().get("detail", [])
    except (ValueError, AttributeError):
        return "Thông tin đăng ký chưa hợp lệ."
    fields = {item.get("loc", [None, None])[-1] for item in detail if isinstance(item, dict)}
    if "password" in fields:
        return "Mật khẩu phải có từ 12 đến 128 ký tự."
    if "email" in fields:
        return "Địa chỉ email không hợp lệ."
    if "display_name" in fields:
        return "Tên hiển thị không hợp lệ."
    return "Thông tin đăng ký chưa hợp lệ."


def remote_session_is_valid():
    """True nếu cookie phiên đang lưu còn được API chấp nhận; False khi API từ chối; None khi không liên lạc được."""
    try:
        response = requests.get(
            f"{API_URL}/auth/me",
            headers=api_user_headers(),
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        return None
    return response.status_code == 200


@app.route("/register", methods=["GET", "POST"])
def register():
    if is_logged_in() and remote_session_is_valid():
        return redirect(url_for("index"))
    if request.method == "GET":
        return render_template("register.html", error=None)
    status = 503
    try:
        response = requests.post(
            f"{API_URL}/auth/register",
            json={
                "email": request.form.get("email", "").strip(),
                "password": request.form.get("password", ""),
                "display_name": request.form.get("display_name", "").strip(),
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        session.clear()
        flash("Tài khoản đã được tạo. Hãy đăng nhập để tiếp tục.", "success")
        return redirect(url_for("login"))
    except requests.HTTPError as exc:
        status = api_error_status(exc, 502)
        if status == 409:
            message = "Email đã được đăng ký."
        elif status == 422:
            message = friendly_register_error(exc)
        else:
            message = "Không tạo được tài khoản. Hãy thử lại."
    except (requests.RequestException, ValueError):
        message = "Không kết nối được máy chủ xác thực. Hãy kiểm tra API đang chạy rồi thử lại."
        status = 503
    return render_template("register.html", error=message), status


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        if is_logged_in():
            valid = remote_session_is_valid()
            if valid:
                return redirect(url_for("index"))
            if valid is False:
                session.clear()
        return render_template("login.html", error=None, next_url=safe_next_url(request.args.get("next")))
    next_url = safe_next_url(request.form.get("next"))
    status = 503
    try:
        _auth_request(
            "/auth/login",
            {
                "email": request.form.get("email", "").strip(),
                "password": request.form.get("password", ""),
            },
        )
        if session["api_user"].get("role") == "admin":
            return redirect(url_for("admin_dashboard"))
        return redirect(next_url or url_for("index"))
    except requests.HTTPError as exc:
        status = api_error_status(exc, 502)
        if status in (401, 422):
            message = "Email hoặc mật khẩu không đúng."
        else:
            message = "Máy chủ xác thực đang gặp lỗi. Hãy thử lại sau."
    except (requests.RequestException, ValueError):
        message = "Không kết nối được máy chủ xác thực. Hãy kiểm tra API đang chạy rồi thử lại."
        status = 503
    return render_template("login.html", error=message, next_url=next_url), status


@app.post("/logout")
def logout():
    token = session.get("api_auth_cookie")
    if token:
        try:
            requests.post(
                f"{API_URL}/auth/logout",
                headers={"Cookie": f"outfit_session={token}"},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException:
            app.logger.warning("API logout không phản hồi; xóa session web cục bộ.")
    session.clear()
    flash("Bạn đã đăng xuất.", "success")
    return redirect(url_for("login"))


@app.get("/")
def index():
    info = get_model_info()
    activities = [dict(name=name, **ACTIVITY_UI.get(name, {"icon": "✦", "description": "Chọn hoạt động hôm nay", "chip": "Hoạt động"})) for name in info["activities"]]
    temperature = session.get("temperature") if session.get("temperature_source") == "manual" else session.get("weather")
    return render_template("select.html", activities=activities, schedule=session.get("schedule"),
                           activity=session.get("activity"), weather=temperature,
                           weather_city=session.get("weather_city", "Hà Nội"), model_available=info["available"],
                           api_error=info["error"], genders=info["genders"])


@app.get("/outfit")
def outfit_start():
    return redirect(url_for("index"))


@app.get("/history")
def history():
    allowed_types = {"viewed", "finalized", "feedback"}
    allowed_ranges = {"day", "week", "month", "all", "from-to"}
    history_type = request.args.get("type", "viewed")
    range_name = request.args.get("range", "all")
    if history_type not in allowed_types or range_name not in allowed_ranges:
        return render_template(
            "history.html",
            history_items=[],
            history_counts={},
            history_type="viewed",
            range_name="all",
            from_date="",
            to_date="",
            total=0,
            page=1,
            page_size=20,
            api_error="Bộ lọc lịch sử không hợp lệ.",
        ), 422
    params = {
        "type": history_type,
        "range": range_name,
        "page": request.args.get("page", "1"),
        "page_size": "20",
    }
    from_date = request.args.get("from", "")
    to_date = request.args.get("to", "")
    if range_name == "from-to":
        params["from"] = from_date
        params["to"] = to_date
    try:
        payload = api_get("/history", params=params, headers=api_user_headers())
        history_items = payload.get("items", [])
        if not isinstance(history_items, list):
            raise ValueError("API trả về lịch sử không đúng định dạng.")
        today_count = None
        try:
            today_payload = api_get(
                "/history",
                params={**params, "range": "day", "page": "1", "page_size": "1"},
                headers=api_user_headers(),
            )
            today_count = int(today_payload.get("total", 0))
        except (requests.RequestException, TypeError, ValueError):
            # Thống kê phụ không được làm mất danh sách lịch sử chính.
            today_count = None
        counts = {history_type: payload.get("total", 0)}
        for item_type in allowed_types - {history_type}:
            count_params = {**params, "type": item_type, "page": "1", "page_size": "1"}
            try:
                count_payload = api_get(
                    "/history", params=count_params, headers=api_user_headers()
                )
                if isinstance(count_payload, dict) and count_payload.get("total") is not None:
                    counts[item_type] = int(count_payload["total"])
            except (requests.RequestException, ValueError, TypeError):
                # Bộ đếm tab là dữ liệu phụ; lỗi của nó không được làm mất danh sách chính.
                continue
        return render_template(
            "history.html",
            history_items=history_items,
            history_counts=counts,
            today_count=today_count,
            history_type=history_type,
            range_name=range_name,
            from_date=from_date,
            to_date=to_date,
            total=payload.get("total", 0),
            page=payload.get("page", 1),
            page_size=payload.get("page_size", 20),
            api_error=None,
        )
    except requests.HTTPError as exc:
        message = "Bạn cần đăng nhập để xem lịch sử." if api_error_status(exc) == 401 else "Không thể tải lịch sử theo bộ lọc đã chọn."
        return render_template(
            "history.html", history_items=[], history_counts={}, history_type=history_type,
            range_name=range_name, from_date=from_date, to_date=to_date, total=0,
            page=1, page_size=20, api_error=message,
        ), api_error_status(exc)
    except (requests.RequestException, ValueError, TypeError):
        return render_template(
            "history.html",
            history_items=[],
            history_counts={},
            history_type=history_type,
            range_name=range_name,
            from_date=from_date,
            to_date=to_date,
            total=0,
            page=1,
            page_size=20,
            api_error="Không tải được lịch sử từ máy chủ gợi ý. Hãy kiểm tra API_URL và thử lại.",
        ), 503


def _history_return_url():
    target = request.form.get("next", "")
    if target.startswith("/") and not target.startswith("//"):
        return target
    return url_for("history")


@app.post("/history/item/<int:request_item_id>/finalize")
def history_finalize(request_item_id):
    endpoint = f"/outfits/{request_item_id}/finalize"
    try:
        if request.form.get("cancel") == "1":
            response = requests.delete(
                f"{API_URL}{endpoint}",
                headers=api_user_headers(),
                timeout=REQUEST_TIMEOUT,
            )
        else:
            response = requests.post(
                f"{API_URL}{endpoint}",
                json={},
                headers=api_user_headers(),
                timeout=REQUEST_TIMEOUT,
            )
        response.raise_for_status()
        flash(
            "Đã hủy chốt trang phục." if request.form.get("cancel") == "1"
            else "Đã chốt trang phục.",
            "success",
        )
    except requests.HTTPError as exc:
        flash(
            "Không tìm thấy bộ trang phục trong lịch sử."
            if api_error_status(exc) == 404
            else "Không thể cập nhật trạng thái chốt.",
            "error",
        )
    except requests.RequestException:
        flash("Không kết nối được máy chủ. Chưa thể cập nhật trạng thái chốt.", "error")
    return redirect(_history_return_url())


@app.post("/history/item/<int:request_item_id>/feedback")
def history_rating(request_item_id):
    try:
        value = int(request.form.get("value", ""))
    except ValueError:
        value = 0
    if value not in {-1, 1}:
        flash("Đánh giá không hợp lệ.", "error")
        return redirect(_history_return_url())
    try:
        response = requests.post(
            f"{API_URL}/outfits/{request_item_id}/feedback",
            json={"value": value},
            headers=api_user_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        flash("Đã cập nhật đánh giá.", "success")
    except requests.HTTPError as exc:
        flash(
            "Không tìm thấy bộ trang phục trong lịch sử."
            if api_error_status(exc) == 404
            else "Không thể lưu đánh giá.",
            "error",
        )
    except requests.RequestException:
        flash("Không kết nối được máy chủ. Chưa thể lưu đánh giá.", "error")
    return redirect(_history_return_url())


@app.route("/profile", methods=["GET", "POST"])
def profile():
    error = None
    if request.method == "POST":
        payload = {
            "display_name": " ".join(request.form.get("display_name", "").split()),
            "gender": request.form.get("gender") or None,
            "preferred_style": " ".join(request.form.get("preferred_style", "").split()) or None,
            "default_context": " ".join(request.form.get("default_context", "").split()) or None,
        }
        try:
            response = requests.patch(
                f"{API_URL}/profile",
                json=payload,
                headers=api_user_headers(),
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            session["api_user"] = response.json().get("profile", session["api_user"])
            flash("Hồ sơ đã được cập nhật.", "success")
            return redirect(url_for("profile"))
        except requests.HTTPError as exc:
            status = api_error_status(exc)
            error = "Tên hiển thị hoặc lựa chọn hồ sơ chưa hợp lệ." if status == 422 else "Không thể cập nhật hồ sơ."
        except (requests.RequestException, ValueError, TypeError):
            status = 503
            error = "Không kết nối được máy chủ hồ sơ. Hãy thử lại."
        if status == 401:
            session.clear()
            flash("Phiên đăng nhập đã hết hạn. Hãy đăng nhập lại.", "error")
            return redirect(url_for("login"))
    try:
        response = api_get("/profile", headers=api_user_headers())
        user_profile = response.get("profile", {})
        finalized = api_get(
            "/history",
            params={"type": "finalized", "range": "all", "page": 1, "page_size": 1},
            headers=api_user_headers(),
        )
        feedback_summary = api_get(
            "/history",
            params={"type": "feedback", "range": "all", "page": 1, "page_size": 1},
            headers=api_user_headers(),
        )
        return render_template(
            "profile.html",
            user_profile=user_profile,
            finalized_count=finalized.get("total", 0),
            feedback_count=feedback_summary.get("total", 0),
            error=error,
            success=None,
        ), 200
    except requests.HTTPError as exc:
        if api_error_status(exc) == 401:
            session.clear()
            flash("Phiên đăng nhập đã hết hạn. Hãy đăng nhập lại.", "error")
            return redirect(url_for("login"))
        return render_template(
            "profile.html", user_profile=session.get("api_user", {}), finalized_count=0,
            feedback_count=0, error="Không thể tải hồ sơ từ máy chủ.", success=None,
        ), api_error_status(exc)
    except (requests.RequestException, ValueError, TypeError):
        return render_template(
            "profile.html", user_profile=session.get("api_user", {}), finalized_count=0,
            feedback_count=0, error="Không kết nối được máy chủ hồ sơ.", success=None,
        ), 503


@app.post("/profile/password")
def change_profile_password():
    try:
        response = requests.post(
            f"{API_URL}/profile/password",
            json={
                "current_password": request.form.get("current_password", ""),
                "new_password": request.form.get("new_password", ""),
            },
            headers=api_user_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        flash("Mật khẩu đã được đổi.", "success")
    except requests.HTTPError as exc:
        status = api_error_status(exc)
        flash(
            "Mật khẩu hiện tại không đúng." if status == 401 else
            "Mật khẩu mới phải có ít nhất 12 ký tự và khác mật khẩu hiện tại.",
            "error",
        )
        if status == 401:
            session.clear()
            return redirect(url_for("login"))
    except requests.RequestException:
        flash("Không kết nối được máy chủ. Chưa thể đổi mật khẩu.", "error")
    return redirect(url_for("profile"))


@app.get("/history/item/<int:request_item_id>")
def history_detail(request_item_id):
    try:
        item = None
        history_type = request.args.get("type")
        if history_type:
            if history_type not in {"viewed", "finalized", "feedback"}:
                return render_template("404.html"), 404
            range_name = request.args.get("range", "all")
            if range_name not in {"day", "week", "month", "all", "from-to"}:
                return render_template("404.html"), 404
            try:
                page = max(1, int(request.args.get("page", "1")))
            except ValueError:
                return render_template("404.html"), 404
            params = {
                "type": history_type,
                "range": range_name,
                "page": page,
                "page_size": 20,
            }
            if range_name == "from-to":
                params["from"] = request.args.get("from", "")
                params["to"] = request.args.get("to", "")
            payload = api_get("/history", params=params, headers=api_user_headers())
            entries = payload.get("items", [])
            if not isinstance(entries, list):
                raise ValueError("API trả về lịch sử không đúng định dạng.")
            item = next(
                (entry for entry in entries
                 if entry.get("request_item_id") == request_item_id),
                None,
            )
        else:
            # Giữ tương thích với đường dẫn lịch sử cũ chưa có tham số phân trang.
            for history_type in ("viewed", "finalized", "feedback"):
                payload = api_get(
                    "/history",
                    params={"type": history_type, "range": "all", "page": 1, "page_size": 100},
                    headers=api_user_headers(),
                )
                item = next(
                    (entry for entry in payload.get("items", [])
                     if entry.get("request_item_id") == request_item_id),
                    None,
                )
                if item:
                    break
        if item is None:
            return render_template("404.html"), 404
        saved_outfit = item.get("outfit")
        if isinstance(saved_outfit, dict):
            outfit = dict(saved_outfit)
            outfit.setdefault("id", item.get("rank", 1))
        else:
            outfit = {
                **item,
                "id": item.get("rank", 1),
                "match_score": round(float(item.get("score", 0)) * 10),
                "explanation": item.get("explanation", ""),
                "recommendation": item.get("recommendation", ""),
                "breakdown": item.get("items", []),
            }
        result_id = str(uuid4())
        cache_outfit_result(result_id, {
            "outfits": [outfit],
            "warnings": item.get("warnings", []),
        })
        session["result_id"] = result_id
        return redirect(url_for("detail", outfit_id=outfit["id"]))
    except requests.HTTPError as exc:
        return render_template("404.html"), api_error_status(exc)
    except (requests.RequestException, ValueError, TypeError):
        return render_template("404.html"), 503


@app.get("/ui/admin/export.csv")
def admin_export():
    if not session.get("api_auth_cookie"):
        return jsonify({"message": "Đăng nhập với tài khoản quản trị để xuất dữ liệu."}), 401
    try:
        response = requests.get(
            f"{API_URL}/admin/export.csv",
            headers=api_user_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return Response(
            response.content,
            status=200,
            content_type=response.headers.get("Content-Type", "text/csv; charset=utf-8"),
            headers={"Content-Disposition": response.headers.get("Content-Disposition", 'attachment; filename="outfitai-admin.csv"')},
        )
    except requests.HTTPError as exc:
        return jsonify({"message": "Bạn không có quyền xuất dữ liệu quản trị."}), api_error_status(exc)
    except requests.RequestException:
        return jsonify({"message": "Không kết nối được máy chủ quản trị."}), 503


@app.get("/admin")
def admin_dashboard():
    if session.get("api_user", {}).get("role") != "admin":
        return render_template(
            "admin.html",
            admin_error="Bạn không có quyền xem trang quản trị.",
        ), 403

    period = request.args.get("period", "week")
    if period not in {"day", "week", "month"}:
        period = "week"

    headers = api_user_headers()
    params = {
        "range": period,
        "page": 1,
        "page_size": 100,
    }

    try:
        stats_response = api_get(
            "/admin/stats",
            headers=headers,
        )
        users_response = api_get(
            "/admin/users",
            params={"limit": 100, "page": 1},
            headers=headers,
        )
        outfits_response = api_get(
            "/admin/outfits",
            params={"limit": 100, "page": 1},
            headers=headers,
        )
        viewed_response = api_get(
            "/history",
            params={**params, "type": "viewed"},
            headers=headers,
        )
        finalized_response = api_get(
            "/history",
            params={**params, "type": "finalized"},
            headers=headers,
        )
        feedback_response = api_get(
            "/history",
            params={**params, "type": "feedback"},
            headers=headers,
        )
        recent_response = api_get(
            "/history",
            params={
                "range": "all",
                "page": 1,
                "page_size": 20,
                "type": "viewed",
            },
            headers=headers,
        )

        stats = stats_response.get("stats", {})
        users = users_response.get("items", [])
        outfits = outfits_response.get("items", [])
        viewed = viewed_response.get("items", [])
        finalized = finalized_response.get("items", [])
        feedback_items = feedback_response.get("items", [])
        recent_activity = recent_response.get("items", [])

        return render_template(
            "admin.html",
            admin_error=None,
            period=period,
            stats=stats,
            users=users,
            outfits=outfits,
            viewed=viewed,
            finalized=finalized,
            feedback_items=feedback_items,
            recent_activity=recent_activity,
        )

    except requests.HTTPError as exc:
        status = api_error_status(exc)

        if status == 401:
            session.clear()
            flash(
                "Phiên đăng nhập đã hết hạn. Hãy đăng nhập lại.",
                "error",
            )
            return redirect(url_for("login"))

        if status == 403:
            message = "Tài khoản này không có quyền quản trị."
        else:
            message = "Không thể tải dữ liệu quản trị."

        app.logger.exception("Lỗi HTTP khi tải trang quản trị")

        return render_template(
            "admin.html",
            admin_error=message,
        ), status

    except (requests.RequestException, ValueError, TypeError) as exc:
        app.logger.exception(
            "Lỗi tải trang quản trị: %s",
            exc,
        )

        return render_template(
            "admin.html",
            admin_error=(
                f"Không tải được dữ liệu quản trị "
                f"({type(exc).__name__}). "
                "Hãy kiểm tra cửa sổ chạy server."
            ),
        ), 503

@app.post("/schedule")
def save_schedule():
    if request.form.get("schedule_mode", "create") == "upload":
        uploaded = request.files.get("schedule_image")
        if not uploaded or not uploaded.filename:
            return redirect(url_for("index", schedule_error="missing"))
        if Path(uploaded.filename).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"} or uploaded.mimetype not in {"image/png", "image/jpeg", "image/webp"}:
            return redirect(url_for("index", schedule_error="format"))
        raw = uploaded.read()
        if not raw or len(raw) > 5 * 1024 * 1024:
            return redirect(url_for("index", schedule_error="missing"))
        try:
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
        except (OSError, UnidentifiedImageError):
            return redirect(url_for("index", schedule_error="format"))
        session["schedule_notice"] = "Ảnh hợp lệ nhưng API hiện chưa cung cấp chức năng đọc lịch. Bạn có thể nhập lịch thủ công."
        return redirect(url_for("index", schedule_error="parser"))
    fields = [request.form.getlist(key) for key in ("day", "class_time", "subject", "room", "kind")]
    entries = [{"day": day.strip(), "time": time.strip(), "subject": subject.strip(), "room": room.strip(), "kind": kind.strip()}
               for day, time, subject, room, kind in zip(*fields) if day.strip() and time.strip() and subject.strip()]
    valid_days = {"Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"}
    valid_kinds = {"LT", "ĐA", "NT", "TT"}
    if not entries or len(entries) > 60 or any(e["day"] not in valid_days or e["kind"] not in valid_kinds or len(e["subject"]) > 100 or len(e["room"]) > 50 for e in entries):
        return redirect(url_for("index", schedule_error="empty"))
    session["schedule"] = {"kind": "entries", "entries": entries}
    return redirect(url_for("index"))


@app.post("/activity")
def choose_activity():
    info = get_model_info()
    activity = request.form.get("activity", "")
    gender = request.form.get("gender", "")
    raw_temperature = request.form.get("temperature", "").strip()
    source = request.form.get("temperature_source", "live")
    if activity not in info["activities"] or gender not in info["genders"] or source not in {"live", "manual"}:
        return redirect(url_for("index", schedule_error="gender" if gender not in info["genders"] else "empty"))
    try:
        temperature = float(raw_temperature)
        if not -10 <= temperature <= 50:
            raise ValueError
    except ValueError:
        return redirect(url_for("index", schedule_error="temperature"))
    profile_data = session.get("profile", {})
    profile_data["gender"] = gender
    session["profile"] = profile_data
    session["activity"] = activity
    session["temperature"] = temperature
    session["temperature_source"] = source
    return redirect(url_for("style_select"))


@app.get("/ui/model-info")
def model_info_proxy():
    try:
        return jsonify(api_get("/model-info"))
    except (requests.RequestException, ValueError):
        return jsonify({"message": "Không kết nối được API mô hình."}), 503


@app.get("/ui/weather")
def current_weather():
    city = request.args.get("city", "Hà Nội").strip()[:80] or "Hà Nội"
    try:
        params = {"city": city}
        for coordinate in ("lat", "lon"):
            value = request.args.get(coordinate)
            if value is not None:
                params[coordinate] = value
        payload = api_get("/weather", params=params, timeout=(3, 12))
        session["weather_city"] = payload.get("city") or city
        session["weather"] = payload.get("temperature_for_model", payload.get("temp_current"))
        alert = payload.get("rain_alert") or {}
        session["rain_probability"] = alert.get("probability")
        return jsonify(payload)
    except (requests.RequestException, ValueError, TypeError):
        return jsonify({"message": "Chưa lấy được thời tiết thật từ API. Bạn có thể nhập nhiệt độ thủ công."}), 503


@app.get("/ui/geocode")
def geocode_proxy():
    query = request.args.get("q", "").strip()[:80]
    if not query:
        return jsonify({"message": "Vui lòng nhập tên thành phố."}), 422
    try:
        return jsonify(api_get("/geocode", params={"q": query}, timeout=(3, 6)))
    except (requests.RequestException, ValueError):
        return jsonify({"message": "Chưa tìm được thành phố từ dịch vụ địa điểm."}), 503


@app.route("/style", methods=["GET", "POST"])
def style_select():
    info = get_model_info()
    error = info["error"] if not info["available"] else None
    if request.method == "POST":
        styles, colors = request.form.getlist("styles"), request.form.getlist("colors")
        try:
            temperature = float(request.form.get("temperature", ""))
            formality = int(request.form.get("formality", "35"))
            if not -10 <= temperature <= 50 or not 0 <= formality <= 100:
                raise ValueError
        except ValueError:
            error = "Nhiệt độ phải từ -10 đến 50°C và mức trang trọng từ 0 đến 100."
        if any(value not in info["styles"] for value in styles) or any(value not in info["accepted_colors"] for value in colors):
            error = "Phong cách hoặc màu sắc không nằm trong danh sách API cho phép."
        if error:
            return render_template("style.html", activity=session.get("activity"), temperature=session.get("temperature"),
                                   weather_city=session.get("weather_city", "Hà Nội"), temperature_source=session.get("temperature_source", "manual"),
                                   selected_styles=styles, selected_colors=colors, formality=request.form.get("formality", 35),
                                   notes=request.form.get("notes", "")[:300], styles=info["styles"], colors=info["colors"], api_error=error,
                                   criteria_saved=session.get("criteria_saved", False)), 503 if not info["available"] else 422
        session["temperature"] = temperature
        session["styles"], session["colors"] = styles, colors
        session["formality"] = formality
        session["notes"] = " ".join(request.form.get("notes", "")[:300].split())
        action = request.form.get("action", "save")
        # Lưu tiêu chí vào session và chuyển thẳng sang phân tích AI trong cùng một lần bấm.
        # Không còn bước trung gian "Tạo và lưu bộ outfit".
        session["criteria_saved"] = True
        return redirect(url_for("loading"))
    saved_temperature = session.get("temperature")
    if saved_temperature is None:
        saved_temperature = session.get("weather")
    return render_template("style.html", activity=session.get("activity"), temperature=saved_temperature,
                           weather_city=session.get("weather_city", "Hà Nội"), temperature_source=session.get("temperature_source", "manual"),
                           selected_styles=session.get("styles", []), selected_colors=session.get("colors", []), formality=session.get("formality", 35),
                           notes=session.get("notes", ""), styles=info["styles"], colors=info["colors"], api_error=error,
                           criteria_saved=session.get("criteria_saved", False))


def request_recommendations():
    summary = ""
    schedule = session.get("schedule", {}).get("entries", [])
    if schedule:
        summary = "; ".join(f"{item['time']} {item['subject']}" for item in schedule)[:300]
    payload = {"gender": session.get("profile", {}).get("gender"), "activity": session.get("activity"),
               "temperature": session.get("temperature"), "styles": session.get("styles", []),
               "colors": session.get("colors", []), "formality": session.get("formality", 35), "top_n": 3,
               "temperature_source": session.get("temperature_source", "manual"), "rain_probability": session.get("rain_probability"),
               "schedule_summary": summary, "notes": session.get("notes", "")}
    response = requests.post(
        f"{API_URL}/recommend",
        json=payload,
        headers=api_user_headers(),
        timeout=RECOMMEND_TIMEOUT,
    )
    response.raise_for_status()
    result = response.json()
    outfits = result.get("outfits") if isinstance(result, dict) else None
    if not isinstance(outfits, list):
        raise ValueError("API trả về dữ liệu gợi ý không đúng định dạng.")
    for number, outfit in enumerate(outfits, 1):
        outfit["id"] = number
        if not isinstance(outfit.get("items"), list):
            outfit["items"] = []
        if not isinstance(outfit.get("breakdown"), list):
            outfit["breakdown"] = outfit["items"]
        outfit["match_score"] = outfit.get("match_score", round(float(outfit.get("score", 0)) * 10))
    result_id = str(uuid4())
    cache_outfit_result(result_id, result)
    session["result_id"] = result_id


@app.route("/loading", methods=["GET", "POST"])
def loading():
    if request.method == "GET":
        if not session.get("activity") or not session.get("profile", {}).get("gender") or session.get("temperature") is None:
            return redirect(url_for("index"))
        return render_template("loading.html", error=None), 200
    session.pop("result_id", None)
    try:
        request_recommendations()
        return redirect(url_for("results"))
    except requests.HTTPError as exc:
        if api_error_status(exc) == 401:
            session.clear()
            flash("Phiên đăng nhập đã hết hạn. Hãy đăng nhập lại.", "error")
            return redirect(url_for("login"))
        return render_template(
            "loading.html",
            error="Máy chủ từ chối yêu cầu gợi ý. Hãy kiểm tra lại thông tin đã chọn.",
        ), api_error_status(exc)
    except requests.Timeout:
        return render_template("loading.html", error="API gợi ý phản hồi quá thời gian. Kiểm tra API rồi thử lại."), 504
    except requests.RequestException:
        return render_template("loading.html", error="Không kết nối được API gợi ý. Kiểm tra API rồi thử lại."), 503
    except (ValueError, TypeError, KeyError):
        return render_template("loading.html", error="API trả về kết quả không hợp lệ. Vui lòng thử lại."), 502


@app.get("/results")
def results():
    result = get_outfit_result(session.get("result_id")) or {}
    return render_template("results.html", outfits=result.get("outfits", []), result=result,
                           api_error=None if result else "Chưa có kết quả gợi ý. Hãy chọn tiêu chí và gửi yêu cầu tới API.")


@app.post("/ui/recommendation-explanations")
def refresh_recommendation_explanations():
    """Same-origin proxy; API derives inputs from the authenticated saved request."""
    result = get_outfit_result(session.get("result_id")) or {}
    request_id = result.get("request_id")
    if not request_id:
        return jsonify({"message": "Chưa có gợi ý để tạo lời giải thích."}), 404
    try:
        response = requests.post(
            f"{API_URL}/recommendations/{request_id}/explanations",
            headers=api_user_headers(),
            timeout=EXPLANATION_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout:
        return jsonify({"message": "Gemini phản hồi quá thời gian; đang giữ lời giải thích theo quy tắc."}), 504
    except requests.ConnectionError:
        return jsonify({"message": "Không kết nối được dịch vụ giải thích; đang giữ lời giải thích theo quy tắc."}), 503
    except requests.HTTPError as exc:
        status = api_error_status(exc, 502)
        if status == 401:
            session.clear()
            return jsonify({"message": "Phiên đăng nhập đã hết hạn. Hãy đăng nhập lại."}), 401
        return jsonify({"message": "Dịch vụ giải thích chưa hoàn tất; đang giữ lời giải thích theo quy tắc."}), status
    except (requests.RequestException, ValueError):
        return jsonify({"message": "Dịch vụ giải thích trả về dữ liệu không hợp lệ; đang giữ lời giải thích theo quy tắc."}), 502
    updates = payload.get("outfits") if isinstance(payload, dict) else None
    if not isinstance(updates, list):
        return jsonify({"message": "Dịch vụ giải thích trả về dữ liệu không hợp lệ; đang giữ lời giải thích theo quy tắc."}), 502
    current_by_id = {
        str(outfit.get("id")): outfit
        for outfit in result.get("outfits", [])
        if isinstance(outfit, dict)
    }
    fields = (
        "explanation_llm", "explanation_structured", "explanation_source",
        "comment_source", "explanation_notice",
    )
    for update in updates:
        if not isinstance(update, dict):
            continue
        outfit = current_by_id.get(str(update.get("id")))
        if outfit is not None:
            for key in fields:
                if key in update:
                    outfit[key] = update[key]
    return jsonify(payload), 200, {"Cache-Control": "no-store"}


@app.get("/detail")
@app.get("/detail/<int:outfit_id>")
def detail(outfit_id=1):
    result = get_outfit_result(session.get("result_id")) or {}
    outfits = result.get("outfits", [])
    outfit = next((item for item in outfits if item.get("id") == outfit_id), None)
    if outfit is None:
        return render_template("404.html"), 404
    analysis_rows = []
    for key, label in (
        ("temp_score", "Phù hợp thời tiết"),
        ("activity_score", "Phù hợp hoàn cảnh"),
        ("style_score", "Phù hợp phong cách"),
        ("preference_score", "Phù hợp sở thích"),
        ("criteria_score", "Phù hợp tiêu chí đã chọn"),
        ("match_score", "Điểm xếp hạng"),
    ):
        raw_value = outfit.get(key)
        if raw_value is None and key == "preference_score":
            continue
        try:
            numeric_value = float(raw_value)
            canonical_value = int(round(numeric_value)) if math.isfinite(numeric_value) else 0
        except (TypeError, ValueError):
            canonical_value = 0
        analysis_rows.append({"label": label, "value": max(0, min(100, canonical_value))})
    return render_template(
        "detail.html",
        outfit=outfit,
        analysis_rows=analysis_rows,
        result_count=len(outfits),
        selected_model=result.get("selected_model"),
        selected_activity=session.get("activity"),
        selected_temperature=session.get("temperature"),
        selected_city=session.get("weather_city"),
        selected_styles=session.get("styles", []),
        explanation=outfit.get("explanation", ""),
        recommendation=outfit.get("recommendation", ""),
        request_id=result.get("request_id", ""),
        result_warnings=result.get("warnings", []),
    )


@app.post("/ui/recommend-preview")
def recommend_preview_proxy():
    """Run an authenticated, non-persistent recommendation preview for the style form."""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"message": "Dữ liệu tiêu chí không hợp lệ."}), 422
    try:
        temperature = float(payload.get("temperature", session.get("temperature", 25)))
        formality = int(payload.get("formality", session.get("formality", 35)))
        if not -10 <= temperature <= 50 or not 0 <= formality <= 100:
            return jsonify({"message": "Nhiệt độ hoặc mức trang trọng ngoài giới hạn."}), 422
        styles = payload.get("styles", [])
        colors = payload.get("colors", [])
        if not isinstance(styles, list) or not isinstance(colors, list):
            return jsonify({"message": "Phong cách và màu sắc phải là danh sách."}), 422
        info = get_model_info()
        if any(value not in info["styles"] for value in styles) or any(value not in info["accepted_colors"] for value in colors):
            return jsonify({"message": "Phong cách hoặc màu sắc không hợp lệ."}), 422
        schedule = session.get("schedule", {}).get("entries", [])
        summary = "; ".join(f"{item['time']} {item['subject']}" for item in schedule)[:300]
        body = {
            "gender": session.get("profile", {}).get("gender") or session.get("api_user", {}).get("gender"),
            "activity": session.get("activity"),
            "temperature": temperature,
            "styles": styles,
            "colors": colors,
            "formality": formality,
            "top_n": 3,
            "temperature_source": session.get("temperature_source", "live"),
            "rain_probability": session.get("rain_probability"),
            "schedule_summary": summary,
            "notes": " ".join(str(payload.get("notes", ""))[:300].split()),
        }
        if not body["gender"] or not body["activity"]:
            return jsonify({"message": "Hãy chọn giới tính và hoạt động trước khi xem gợi ý."}), 422
        response = requests.post(
            f"{API_URL}/recommend/preview", json=body, headers=api_user_headers(), timeout=(5, 35)
        )
        response.raise_for_status()
        return jsonify(response.json())
    except requests.HTTPError as exc:
        status = api_error_status(exc, 502)
        message = "Phiên đăng nhập hết hạn." if status == 401 else "API chưa tạo được bản xem trước."
        return jsonify({"message": message}), status
    except requests.Timeout:
        return jsonify({"message": "AI đang phản hồi chậm. Vui lòng đợi hoặc bấm gợi ý chính thức."}), 504
    except requests.RequestException:
        return jsonify({"message": "Không kết nối được API gợi ý."}), 503
    except (ValueError, TypeError, KeyError):
        return jsonify({"message": "Không thể đọc tiêu chí xem trước."}), 422


@app.post("/ui/events/view")
def view_event_proxy():
    if not session.get("api_auth_cookie"):
        return jsonify({"message": "Đăng nhập để ghi nhận lượt xem."}), 401
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"message": "Dữ liệu lượt xem không hợp lệ."}), 422
    try:
        response = requests.post(
            f"{API_URL}/events/view",
            json=payload,
            headers=api_user_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except requests.HTTPError as exc:
        return jsonify({"message": "Chưa cập nhật được lượt xem."}), (
            exc.response.status_code if exc.response is not None else 502
        )
    except requests.RequestException:
        return jsonify({"message": "Không kết nối được máy chủ gợi ý."}), 503


def _result_outfit(display_id):
    result = get_outfit_result(session.get("result_id")) or {}
    return next(
        (item for item in result.get("outfits", []) if item.get("id") == display_id),
        None,
    )


@app.post("/ui/outfits/<int:display_id>/finalize")
def finalize_outfit_proxy(display_id):
    outfit = _result_outfit(display_id)
    if not outfit or not outfit.get("request_item_id"):
        return jsonify({"message": "Không tìm thấy bộ trang phục."}), 404
    try:
        response = requests.post(
            f"{API_URL}/outfits/{outfit['request_item_id']}/finalize",
            json={"note": request.form.get("note", "")[:500] or None},
            headers=api_user_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except requests.HTTPError as exc:
        return jsonify({"message": "Chưa chốt được trang phục."}), (
            exc.response.status_code if exc.response is not None else 502
        )
    except requests.RequestException:
        return jsonify({"message": "Không kết nối được máy chủ gợi ý."}), 503


@app.delete("/ui/outfits/<int:display_id>/finalize")
def cancel_finalize_outfit_proxy(display_id):
    outfit = _result_outfit(display_id)
    if not outfit or not outfit.get("request_item_id"):
        return jsonify({"message": "Không tìm thấy bộ trang phục."}), 404
    try:
        response = requests.delete(
            f"{API_URL}/outfits/{outfit['request_item_id']}/finalize",
            headers=api_user_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except requests.HTTPError as exc:
        return jsonify({"message": "Chưa thể hủy chốt trang phục."}), (
            exc.response.status_code if exc.response is not None else 502
        )
    except requests.RequestException:
        return jsonify({"message": "Không kết nối được máy chủ gợi ý."}), 503


@app.post("/ui/outfits/<int:display_id>/feedback")
def outfit_feedback_proxy(display_id):
    outfit = _result_outfit(display_id)
    if not outfit or not outfit.get("request_item_id"):
        return jsonify({"message": "Không tìm thấy bộ trang phục."}), 404
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or payload.get("value") not in {-1, 1}:
        return jsonify({"message": "Đánh giá chỉ nhận hữu ích hoặc không hữu ích."}), 422
    try:
        response = requests.post(
            f"{API_URL}/outfits/{outfit['request_item_id']}/feedback",
            json={"value": payload["value"], "reason": payload.get("reason")},
            headers=api_user_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except requests.HTTPError as exc:
        return jsonify({"message": "Chưa lưu được đánh giá."}), (
            exc.response.status_code if exc.response is not None else 502
        )
    except requests.RequestException:
        return jsonify({"message": "Không kết nối được máy chủ gợi ý."}), 503


@app.post("/feedback")
def feedback():
    action = request.form.get("action", "")
    request_id = request.form.get("request_id", "")
    try:
        outfit_id = int(request.form.get("outfit_id", ""))
    except ValueError:
        flash("Không xác định được bộ trang phục cần phản hồi.", "error")
        return redirect(url_for("results"))
    if action not in {"worn", "like", "dislike"} or not request_id:
        flash("Phản hồi không hợp lệ.", "error")
        return redirect(url_for("detail", outfit_id=outfit_id))
    note = " ".join(request.form.get("note", "").split())[:300] or None
    try:
        response = requests.post(
            f"{API_URL}/feedback",
            json={
                "request_id": request_id,
                "outfit_id": outfit_id,
                "action": action,
                "note": note,
            },
            headers=api_user_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        message = {
            "worn": "Đã ghi nhận bạn chọn mặc bộ này.",
            "like": "Đã ghi nhận đánh giá thích của bạn.",
            "dislike": "Đã ghi nhận đánh giá chưa phù hợp.",
        }[action]
        flash(message, "success")
    except requests.RequestException:
        flash("Chưa lưu được phản hồi. Hãy kiểm tra API và thử lại.", "error")
    return redirect(url_for("detail", outfit_id=outfit_id))


@app.errorhandler(404)
def page_not_found(_):
    return render_template("404.html"), 404


@app.errorhandler(413)
def file_too_large(_):
    return "Ảnh lịch học vượt quá giới hạn 5 MB.", 413


if __name__ == "__main__":
    app.run(
        debug=os.environ.get("FLASK_DEBUG") == "1",
        host=os.environ.get("FLASK_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASK_PORT", "5000")),
    )
