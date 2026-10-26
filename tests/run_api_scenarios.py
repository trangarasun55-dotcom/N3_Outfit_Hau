"""Run measurable API scenarios against an isolated, temporary SQLite database."""

from __future__ import annotations

import os
import secrets
import time
from typing import Any, Callable
from uuid import UUID

from test_db_utils import isolated_test_db


class ScenarioRunner:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, float, bool]] = []

    def run(
        self,
        name: str,
        expected: str,
        action: Callable[[], tuple[bool, str]],
    ) -> Any:
        started = time.perf_counter()
        try:
            passed, actual = action()
        except Exception as exc:
            passed, actual = False, f"{type(exc).__name__}: {exc}"
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.rows.append((name, expected, actual, elapsed_ms, passed))
        return passed

    def report(self) -> bool:
        print("| Trường hợp | Mong đợi | Thực tế | Thời gian |")
        print("|---|---|---|---:|")
        for name, expected, actual, elapsed_ms, passed in self.rows:
            status = "OK" if passed else "FAIL"
            print(
                f"| {name} | {expected} | {status}: {actual} | {elapsed_ms:.1f} ms |"
            )
        passed_count = sum(row[-1] for row in self.rows)
        print(f"\nKết quả: {passed_count}/{len(self.rows)} kịch bản đạt.")
        return passed_count == len(self.rows)


def main() -> int:
    with isolated_test_db("scenarios") as db_path:
        os.environ["DB_PATH"] = str(db_path)
        os.environ["GEMINI_API_KEY"] = ""
        os.environ["ADMIN_EMAIL"] = "scenario-admin@example.test"
        os.environ["ADMIN_PASSWORD"] = secrets.token_urlsafe(28)
        os.environ["SECRET_KEY"] = secrets.token_urlsafe(48)

        from fastapi.testclient import TestClient

        from api import main as api
        from api import storage

        runner = ScenarioRunner()
        user_client = TestClient(api.app)
        admin_client = TestClient(api.app)
        with user_client, admin_client:
            runner.run(
                "health",
                "HTTP 200, status=ok",
                lambda: (
                    (response := user_client.get("/health")).status_code == 200
                    and response.json().get("status") == "ok",
                    f"HTTP {response.status_code}",
                ),
            )
            runner.run(
                "model-info",
                "HTTP 200, có metadata categories",
                lambda: (
                    (response := user_client.get("/model-info")).status_code == 200
                    and bool(response.json().get("categories")),
                    f"HTTP {response.status_code}",
                ),
            )
            predict_payload = {
                "Hoàn cảnh": "Xưởng/Công trình",
                "Phong cách": "Streetwear",
                "Tmin (°C)": 20,
                "Tmax (°C)": 38,
                "Nhiệt độ ngày (°C)": 12,
                "Giữ ấm": 2,
                "Thoáng khí": 5,
                "Vận động": 2,
                "Chỉn chu": 2,
            }
            runner.run(
                "predict hợp lệ",
                "HTTP 200, điểm trong 0..10",
                lambda: (
                    (response := user_client.post("/predict", json=predict_payload)).status_code == 200
                    and 0 <= response.json().get("score", -1) <= 10,
                    f"HTTP {response.status_code}",
                ),
            )
            runner.run(
                "predict thiếu trường",
                "HTTP 422",
                lambda: (
                    (response := user_client.post("/predict", json={})).status_code == 422,
                    f"HTTP {response.status_code}",
                ),
            )
            runner.run(
                "recommend chưa đăng nhập",
                "HTTP 401",
                lambda: (
                    (response := user_client.post(
                        "/recommend",
                        json={"gender": "Nữ", "activity": "Đi học", "temperature": 22},
                    )).status_code == 401,
                    f"HTTP {response.status_code}",
                ),
            )

            email = f"scenario-{secrets.token_hex(6)}@example.test"
            password = secrets.token_urlsafe(24)
            register = user_client.post(
                "/auth/register",
                json={
                    "email": email,
                    "password": password,
                    "display_name": "Scenario User",
                },
            )
            runner.run(
                "đăng ký user",
                "HTTP 201",
                lambda: (register.status_code == 201, f"HTTP {register.status_code}"),
            )
            user_client.post("/auth/logout")
            login = user_client.post(
                "/auth/login", json={"email": email, "password": password}
            )
            runner.run(
                "đăng nhập user",
                "HTTP 200, cấp phiên",
                lambda: (
                    login.status_code == 200
                    and bool(user_client.cookies.get("outfit_session")),
                    f"HTTP {login.status_code}",
                ),
            )
            runner.run(
                "user bị chặn admin",
                "HTTP 403",
                lambda: (
                    (response := user_client.get("/admin/stats")).status_code == 403,
                    f"HTTP {response.status_code}",
                ),
            )

            recommendation_payload = {
                "gender": "Nữ",
                "activity": "Đi học",
                "temperature": 22,
                "styles": [],
                "colors": [],
                "top_n": 3,
            }
            recommendation: dict[str, Any] = {}

            def recommend() -> tuple[bool, str]:
                nonlocal recommendation
                response = user_client.post(
                    "/recommend", json=recommendation_payload
                )
                recommendation = response.json() if response.status_code == 200 else {}
                outfits = recommendation.get("outfits", [])
                unique_compositions = {
                    tuple(sorted(str(item.get("id")) for item in outfit.get("items", [])))
                    for outfit in outfits
                }
                comments = [
                    outfit.get("explanation_llm") or outfit.get("explanation")
                    for outfit in outfits
                ]
                valid = (
                    response.status_code == 200
                    and len(outfits) == 3
                    and len(unique_compositions) == 3
                    and len(set(comments)) == 3
                    and all(float(outfit.get("score", 0)) >= 6 for outfit in outfits)
                )
                scores = [float(outfit.get("score", 0)) for outfit in outfits]
                return valid, (
                    f"HTTP {response.status_code}; bộ={len(outfits)}/3; "
                    f"thành phần khác nhau={len(unique_compositions)}/3; "
                    f"lý do khác nhau={len(set(comments))}/3; "
                    f"điểm thấp nhất={min(scores, default=0):.2f}/6"
                )

            runner.run(
                "recommend top 3",
                "HTTP 200, 3 bộ khác nhau, điểm >= 6",
                recommend,
            )

            def explain_top_three() -> tuple[bool, str]:
                request_id = recommendation.get("request_id")
                if not request_id:
                    return False, "recommend không trả request_id"
                response = user_client.post(
                    f"/recommendations/{request_id}/explanations"
                )
                outfits = response.json().get("outfits", [])
                valid = (
                    response.status_code == 200
                    and len(outfits) == 3
                    and all(item.get("explanation_source") == "rules" for item in outfits)
                    and all(item.get("explanation_llm") for item in outfits)
                )
                return valid, f"HTTP {response.status_code}; có lý do dự phòng={len(outfits)}/3"

            runner.run(
                "giải thích top 3 tách biệt",
                "HTTP 200, đủ 3 lý do theo quy tắc khi Gemini chưa cấu hình",
                explain_top_three,
            )

            first_outfit = (
                recommendation.get("outfits", [{}])[0]
                if recommendation.get("outfits")
                else {}
            )
            request_item_id = first_outfit.get("request_item_id")
            view: dict[str, Any] = {}
            if request_item_id:
                def start_view() -> tuple[bool, str]:
                    nonlocal view
                    response = user_client.post(
                        "/events/view",
                        json={"action": "start", "request_item_id": request_item_id},
                    )
                    view = response.json() if response.status_code == 200 else {}
                    return response.status_code == 200, f"HTTP {response.status_code}"

                runner.run("bắt đầu lượt xem", "HTTP 200", start_view)
                runner.run(
                    "chốt outfit",
                    "HTTP 200",
                    lambda: (
                        (response := user_client.post(
                            f"/outfits/{request_item_id}/finalize", json={}
                        )).status_code == 200,
                        f"HTTP {response.status_code}",
                    ),
                )
                runner.run(
                    "đánh giá outfit",
                    "HTTP 200, ghi hữu ích",
                    lambda: (
                        (response := user_client.post(
                            f"/outfits/{request_item_id}/feedback",
                            json={"value": 1, "reason": "Kịch bản kiểm tra"},
                        )).status_code == 200,
                        f"HTTP {response.status_code}",
                    ),
                )
                runner.run(
                    "lịch sử cá nhân",
                    "HTTP 200, có mục đã chốt",
                    lambda: (
                        (response := user_client.get(
                            "/history?type=finalized&range=all"
                        )).status_code == 200
                        and response.json().get("total", 0) == 1,
                        f"HTTP {response.status_code}",
                    ),
                )
            else:
                for name in (
                    "bắt đầu lượt xem",
                    "chốt outfit",
                    "đánh giá outfit",
                    "lịch sử cá nhân",
                ):
                    runner.rows.append(
                        (name, "chỉ chạy sau recommend thành công", "FAIL: thiếu outfit", 0, False)
                    )

            admin_login = admin_client.post(
                "/auth/login",
                json={
                    "email": os.environ["ADMIN_EMAIL"],
                    "password": os.environ["ADMIN_PASSWORD"],
                },
            )
            runner.run(
                "đăng nhập admin seed",
                "HTTP 200, role=admin",
                lambda: (
                    admin_login.status_code == 200
                    and admin_login.json().get("user", {}).get("role") == "admin",
                    f"HTTP {admin_login.status_code}",
                ),
            )
            for path, name in (
                ("/admin/stats", "admin thống kê"),
                ("/admin/users", "admin danh sách user"),
                ("/admin/outfits", "admin top outfit"),
                ("/admin/export.csv", "admin xuất CSV"),
            ):
                runner.run(
                    name,
                    "HTTP 200",
                    lambda path=path: (
                        (response := admin_client.get(path)).status_code == 200,
                        f"HTTP {response.status_code}",
                    ),
                )

        stats_after_restart = storage.admin_stats()
        runner.run(
            "dữ liệu tồn tại sau phiên API",
            "DB còn user và lượt gợi ý",
            lambda: (
                stats_after_restart["total_users"] >= 1
                and stats_after_restart["total_recommendations"] >= 1,
                f"users={stats_after_restart['total_users']}, recommendations={stats_after_restart['total_recommendations']}",
            ),
        )
        return 0 if runner.report() else 1


if __name__ == "__main__":
    raise SystemExit(main())
