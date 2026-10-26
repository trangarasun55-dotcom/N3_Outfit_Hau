"""Structured Gemini comments, validation, cache and deterministic fallbacks."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from api import llm_explain


REQUEST = {
    "gender": "Nữ",
    "activity": "Đi học",
    "temperature": 22,
    "temperature_source": "live",
    "styles": ["Casual"],
    "colors": ["Đen"],
    "formality": 35,
    "rain_probability": 45,
    "schedule_summary": "07:30 Thiết kế tại A1",
}
RESULT = {
    "outfits": [
        {
            "id": 1,
            "name": "Casual – Đi học",
            "score": 8.1,
            "score_low": 7.5,
            "score_high": 8.7,
            "confidence_level": "Cao",
            "items": [
                {
                    "name": "Áo sơ mi trắng",
                    "score": 8.1,
                    "temperature_range": {"min": 18.0, "max": 28.0},
                    "characteristics": {"warmth": 2.5, "breathability": 4.0},
                }
            ],
            "explanation": "Giải thích theo quy tắc.",
        }
    ]
}
THREE_RESULT = {
    "outfits": [
        {
            "id": index,
            "name": names[index - 1],
            "score": scores[index - 1],
            "score_low": scores[index - 1] - 0.5,
            "score_high": scores[index - 1] + 0.5,
            "confidence_level": "Cao",
            "temp_score": 90,
            "activity_score": 80,
            "style_score": 70,
            "match_score": 82,
            "items": [
                {
                    "name": item_names[index - 1],
                    "score": scores[index - 1],
                    "temperature_range": {"min": 18, "max": 30},
                    "characteristics": {
                        "warmth": 2.5,
                        "breathability": 4,
                        "mobility": 4,
                        "formality": 3.5,
                    },
                }
            ],
            "explanation": f"Lời giải thích quy tắc số {index}.",
        }
        for index in range(1, 4)
        for names in [["Casual – Đi học #1", "Casual – Đi học #2", "Casual – Đi học #3"]]
        for scores in [[8.1, 7.8, 7.5]]
        for item_names in [["Áo sơ mi trắng", "Áo len mỏng", "Áo khoác dáng ngắn"]]
    ]
}


def _structured_payload():
    return {
        "explanations": [
            {
                "rank": 1,
                "headline": "Phương án sáng gọn",
                "why_suitable": "**Áo sơ mi trắng** tạo điểm nhấn rõ cho combo Đi học này. Mô hình chấm tổng thể 8.1/10; dữ liệu của áo ghi độ thoáng khí 4/5, một căn cứ cụ thể để cân nhắc khi bạn ưu tiên độ thoáng.",
                "weather_fit": "Đi học ở 22°C nằm trong khoảng nhiệt của món áo.",
                "caution": "Độ tin cậy của bộ này là Cao.",
                "tip": "Có thể phối áo sơ mi trắng theo lựa chọn gọn nhẹ.",
            },
            {
                "rank": 2,
                "headline": "Lựa chọn mềm mại",
                "why_suitable": "**Áo len mỏng** là điểm khác biệt của combo Đi học này. Mô hình chấm tổng thể 7.8/10; dữ liệu ghi mức giữ ấm 2.5/5, vì vậy bộ này không thiên về khả năng giữ nhiệt.",
                "weather_fit": "Ở 22°C, khoảng nhiệt của bộ cho phép lựa chọn này.",
                "caution": "Mức tin cậy hiện tại là Cao.",
                "tip": "Thử lấy áo len mỏng làm món bắt đầu phối.",
            },
            {
                "rank": 3,
                "headline": "Tổ hợp có điểm nhấn",
                "why_suitable": "**Áo khoác dáng ngắn** là món đáng chú ý trong combo Đi học này. Điểm model tổng thể là 7.5/10; dữ liệu ghi mức linh hoạt 4/5, giúp bạn cân nhắc bộ này khi hoạt động cần di chuyển.",
                "weather_fit": "Nhiệt độ 22°C nằm trong giới hạn đã dùng để chọn món.",
                "caution": "Kết quả được đánh giá với độ tin cậy Cao.",
                "tip": "Chọn áo khoác dáng ngắn làm điểm khởi đầu của cách phối.",
            },
        ]
    }


def _client_mock(responses):
    client = Mock()
    client.models.generate_content.side_effect = [
        Mock(text=json.dumps(response, ensure_ascii=False)) for response in responses
    ]
    context_client = Mock()
    context_client.__enter__ = Mock(return_value=client)
    context_client.__exit__ = Mock(return_value=False)
    return client, context_client


class LlmExplainTests(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(
            os.environ, {"GEMINI_API_KEY": "", "GEMINI_MODEL": "test-model"}
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        with llm_explain._cache_lock:
            llm_explain._cache.clear()
        with llm_explain._fallback_model_cache_lock:
            llm_explain._fallback_model_cache.clear()

    def test_missing_api_key_returns_rules_fallback(self):
        result = json.loads(json.dumps(RESULT))
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            llm_explain.add_explanations(result, REQUEST)
        self.assertEqual(result["outfits"][0]["explanation_llm"], "Giải thích theo quy tắc.")
        self.assertEqual(result["outfits"][0]["explanation_source"], "rules")
        self.assertIn("chưa cấu hình GEMINI_API_KEY", result["outfits"][0]["explanation_notice"])

    def test_three_structured_explanations_are_validated_and_cached(self):
        result = json.loads(json.dumps(THREE_RESULT))
        client, context_client = _client_mock([_structured_payload()])
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-test"}),
            patch("api.llm_explain.genai.Client", return_value=context_client) as client_factory,
        ):
            llm_explain.add_explanations(result, REQUEST)
            second = json.loads(json.dumps(THREE_RESULT))
            llm_explain.add_explanations(second, REQUEST)
        self.assertTrue(all(item["explanation_source"] == "gemini" for item in result["outfits"]))
        self.assertTrue(all(item["comment_source"] == "gemini" for item in result["outfits"]))
        self.assertTrue(all(item["explanation_structured"] for item in result["outfits"]))
        self.assertEqual(len({item["explanation_llm"] for item in result["outfits"]}), 3)
        self.assertTrue(
            all(item["explanation_structured"].get("weather_fit") for item in result["outfits"])
        )
        self.assertTrue(
            all(item["explanation_structured"].get("vs_others") for item in result["outfits"])
        )
        self.assertTrue(
            result["outfits"][0]["explanation_structured"]["headline"].startswith("Ưu tiên:")
        )
        self.assertEqual(client_factory.call_count, 1)
        client.models.generate_content.assert_called_once()
        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertGreaterEqual(
            client_factory.call_args.kwargs["http_options"].timeout, 10_000
        )

    def test_unavailable_model_is_not_replaced_with_a_guessed_model(self):
        result = json.loads(json.dumps(THREE_RESULT))
        client = Mock()

        class ModelGone(Exception):
            code = 404

        client.models.generate_content.side_effect = ModelGone("model not found")
        context_client = Mock()
        context_client.__enter__ = Mock(return_value=client)
        context_client.__exit__ = Mock(return_value=False)
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "model-da-ngung"}),
            patch("api.llm_explain.genai.Client", return_value=context_client),
        ):
            llm_explain.add_explanations(result, REQUEST)
        client.models.generate_content.assert_called_once()
        self.assertEqual(
            client.models.generate_content.call_args.kwargs["model"], "model-da-ngung"
        )
        self.assertTrue(all(item["explanation_source"] == "rules" for item in result["outfits"]))
        self.assertTrue(
            all("GEMINI_MODEL" in item["explanation_notice"] for item in result["outfits"])
        )

    def test_provider_failures_show_safe_actionable_notice(self):
        cases = [
            (401, "từ chối xác thực"),
            (403, "từ chối xác thực"),
            (404, "không tìm thấy model"),
            (429, "vượt hạn mức"),
            (504, "quá thời gian chờ"),
        ]
        for status, expected_notice in cases:
            with self.subTest(status=status):
                error = type(
                    f"ProviderError{status}",
                    (Exception,),
                    {"code": status},
                )("SENSITIVE_PROVIDER_DETAIL")
                notice = llm_explain._provider_failure_notice(error)
                self.assertIn(expected_notice, notice)
                self.assertNotIn("SENSITIVE_PROVIDER_DETAIL", notice)

        connection_error = type("ConnectError", (Exception,), {})(
            "SENSITIVE_PROVIDER_DETAIL"
        )
        self.assertIn("không kết nối được", llm_explain._provider_failure_notice(connection_error))
        self.assertFalse(llm_explain._should_retry_provider_error(connection_error))
        self.assertTrue(
            llm_explain._should_retry_provider_error(
                type("ProviderError503", (Exception,), {"code": 503})()
            )
        )
        self.assertFalse(
            llm_explain._should_retry_provider_error(
                type("ProviderError504", (Exception,), {"code": 504})()
            )
        )
        self.assertFalse(
            llm_explain._should_retry_provider_error(
                type("ProviderError401", (Exception,), {"code": 401})()
            )
        )

    def test_fallback_model_discovery_uses_only_available_text_flash_models(self):
        models = [
            SimpleNamespace(
                name="models/gemini-3.5-flash",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(
                name="models/gemini-3.5-flash-lite",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(
                name="models/gemini-3.6-flash-image-preview",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(
                name="models/gemini-4-pro",
                supported_actions=["generateContent"],
            ),
            SimpleNamespace(
                name="models/gemini-3.8-flash-tts",
                supported_actions=["generateContent"],
            ),
        ]
        client, context_client = _client_mock([])
        client.models.list.return_value = models
        with patch("api.llm_explain.genai.Client", return_value=context_client):
            candidates = llm_explain._fallback_models("test-key", "models/gemini-primary")
        self.assertEqual(candidates, ("models/gemini-3.5-flash-lite", "models/gemini-3.5-flash"))

    def test_transient_provider_error_retries_with_api_discovered_model(self):
        result = json.loads(json.dumps(THREE_RESULT))
        client, context_client = _client_mock([_structured_payload()])

        class ServiceUnavailable(Exception):
            code = 503

        client.models.generate_content.side_effect = [
            ServiceUnavailable("SENSITIVE_PROVIDER_DETAIL"),
            Mock(text=json.dumps(_structured_payload(), ensure_ascii=False)),
        ]
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-primary"}),
            patch("api.llm_explain.genai.Client", return_value=context_client),
            patch("api.llm_explain._fallback_models", return_value=("models/gemini-alternate",)),
        ):
            llm_explain.add_explanations(result, REQUEST)
        self.assertEqual(client.models.generate_content.call_count, 2)
        self.assertEqual(
            [call.kwargs["model"] for call in client.models.generate_content.call_args_list],
            ["gemini-primary", "models/gemini-alternate"],
        )
        self.assertTrue(all(item["explanation_source"] == "gemini" for item in result["outfits"]))

    def test_timeout_after_invalid_content_reports_provider_error_not_validation_error(self):
        result = json.loads(json.dumps(THREE_RESULT))
        duplicate = _structured_payload()
        duplicate["explanations"][0]["why_suitable"] = "Câu không hợp lệ."
        client, context_client = _client_mock([duplicate])

        class GatewayTimeout(Exception):
            code = 504

        client.models.generate_content.side_effect = [
            Mock(text=json.dumps(duplicate, ensure_ascii=False)),
            GatewayTimeout("SENSITIVE_PROVIDER_DETAIL"),
        ]
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-test"}),
            patch("api.llm_explain.genai.Client", return_value=context_client),
        ):
            llm_explain.add_explanations(result, REQUEST)
        self.assertEqual(client.models.generate_content.call_count, 2)
        self.assertTrue(
            all("quá thời gian chờ" in item["explanation_notice"] for item in result["outfits"])
        )
        self.assertTrue(
            all("kiểm tra dữ kiện" not in item["explanation_notice"] for item in result["outfits"])
        )

    def test_authentication_failure_notice_reaches_user_without_provider_details(self):
        result = json.loads(json.dumps(THREE_RESULT))
        error = type("AuthError", (Exception,), {"code": 401})(
            "SENSITIVE_PROVIDER_DETAIL"
        )
        context_client = Mock()
        context_client.__enter__ = Mock(side_effect=error)
        context_client.__exit__ = Mock(return_value=False)
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-test"}),
            patch("api.llm_explain.genai.Client", return_value=context_client),
        ):
            llm_explain.add_explanations(result, REQUEST)
        self.assertTrue(
            all("từ chối xác thực" in item["explanation_notice"] for item in result["outfits"])
        )
        self.assertTrue(
            all("SENSITIVE_PROVIDER_DETAIL" not in item["explanation_notice"] for item in result["outfits"])
        )

    def test_rounded_score_is_accepted_but_invented_number_is_rejected(self):
        result = json.loads(json.dumps(THREE_RESULT))
        result["outfits"][0]["score"] = 8.1234
        payload = _structured_payload()
        payload["explanations"][0]["why_suitable"] = (
            "**Áo sơ mi trắng** tạo điểm nhấn cho combo Đi học này. Mô hình chấm 8.12/10; "
            "dữ liệu ghi độ thoáng khí 4/5 để bạn cân nhắc khi ưu tiên độ thoáng."
        )
        llm_explain._validate_generated(
            {int(entry["rank"]): entry for entry in payload["explanations"]},
            REQUEST,
            result["outfits"],
        )
        payload["explanations"][0]["why_suitable"] += " Giảm giá 37%."
        with self.assertRaises(ValueError):
            llm_explain._validate_generated(
                {int(entry["rank"]): entry for entry in payload["explanations"]},
                REQUEST,
                result["outfits"],
            )

    def test_reason_accepts_up_to_80_words_and_rejects_81(self):
        payload = _structured_payload()
        generated = {int(entry["rank"]): entry for entry in payload["explanations"]}
        first_sentence, second_sentence = generated[1]["why_suitable"].split(". ", 1)
        second_sentence = second_sentence.rstrip(".")
        while len(llm_explain._WORD_RE.findall(generated[1]["why_suitable"])) < 80:
            second_sentence += " thêm"
            generated[1]["why_suitable"] = f"{first_sentence}. {second_sentence}."
        llm_explain._validate_generated(generated, REQUEST, THREE_RESULT["outfits"])
        generated[1]["why_suitable"] = f"{first_sentence}. {second_sentence} thêm."
        with self.assertRaisesRegex(ValueError, "explanation length"):
            llm_explain._validate_generated(generated, REQUEST, THREE_RESULT["outfits"])

    def test_internal_feature_names_are_rendered_in_vietnamese(self):
        translated = llm_explain._translate_feature_names(
            "breathability 4/5, warmth 2.5/5, mobility 3/5, formality 4/5"
        )
        self.assertEqual(
            translated,
            "độ thoáng khí 4/5, độ giữ ấm 2.5/5, mức linh hoạt 3/5, mức chỉn chu 4/5",
        )

    def test_missing_activity_is_added_from_request_without_recalling_gemini(self):
        payload = _structured_payload()
        payload["explanations"][0]["why_suitable"] = payload["explanations"][0][
            "why_suitable"
        ].replace("combo Đi học này", "bộ trang phục này")
        generated = {int(entry["rank"]): entry for entry in payload["explanations"]}

        llm_explain._validate_generated(generated, REQUEST, THREE_RESULT["outfits"])

        self.assertIn("trong bối cảnh đi học", generated[1]["why_suitable"])

    def test_translated_item_feature_reasons_pass_fact_validation(self):
        reasons = [
            "**Áo sơ mi trắng** tạo điểm nhấn cho combo Đi học này. Mô hình chấm 8.1/10; dữ liệu ghi breathability 4/5 để bạn cân nhắc khi ưu tiên độ thoáng.",
            "**Áo len mỏng** là điểm khác biệt của combo Đi học này. Mô hình chấm 7.8/10; dữ liệu ghi warmth 2.5/5, vì vậy bộ này không thiên về khả năng giữ nhiệt.",
            "**Áo khoác dáng ngắn** là món đáng chú ý trong combo Đi học này. Điểm model là 7.5/10; dữ liệu ghi mobility 4/5, giúp bạn cân nhắc bộ này khi hoạt động cần di chuyển.",
        ]
        generated = {
            index: {
                "rank": index,
                "why_suitable": llm_explain._translate_feature_names(reason),
            }
            for index, reason in enumerate(reasons, start=1)
        }
        llm_explain._validate_generated(generated, REQUEST, THREE_RESULT["outfits"])

    def test_emphasis_may_highlight_a_real_part_of_the_full_item_name(self):
        outfits = json.loads(json.dumps(THREE_RESULT["outfits"]))
        outfits[0]["items"][0]["name"] = "Áo sơ mi trắng form suông"
        payload = _structured_payload()
        payload["explanations"][0]["why_suitable"] = (
            "**Áo sơ mi trắng** tạo điểm nhấn cho combo Đi học này. Mô hình chấm 8.1/10; "
            "dữ liệu ghi độ thoáng khí 4/5 để bạn cân nhắc khi ưu tiên độ thoáng."
        )
        generated = {int(entry["rank"]): entry for entry in payload["explanations"]}

        llm_explain._validate_generated(generated, REQUEST, outfits)

    def test_unlisted_clothing_item_in_gemini_reason_is_rejected(self):
        payload = _structured_payload()
        payload["explanations"][0]["why_suitable"] += (
            " Nên kết hợp thêm quần âu."
        )
        with self.assertRaisesRegex(ValueError, "clothing item absent"):
            llm_explain._validate_generated(
                {int(entry["rank"]): entry for entry in payload["explanations"]},
                REQUEST,
                THREE_RESULT["outfits"],
            )

    def test_material_claim_is_allowed_only_when_present_in_outfit_facts(self):
        outfits = json.loads(json.dumps(THREE_RESULT["outfits"]))
        outfits[0]["items"][0]["material"] = "cotton"
        payload = _structured_payload()
        payload["explanations"][0]["why_suitable"] = (
            "**Áo sơ mi trắng** tạo điểm nhấn cho combo Đi học này. Mô hình chấm 8.1/10; "
            "dữ liệu ghi độ thoáng khí 4/5 và chất liệu cotton để bạn cân nhắc lựa chọn."
        )
        generated = {int(entry["rank"]): entry for entry in payload["explanations"]}
        llm_explain._validate_generated(generated, REQUEST, outfits)
        outfits[0]["items"][0].pop("material")
        with self.assertRaisesRegex(ValueError, "unlisted clothing material"):
            llm_explain._validate_generated(generated, REQUEST, outfits)

    def test_unverified_garment_care_advice_is_rejected(self):
        payload = _structured_payload()
        payload["explanations"][1]["why_suitable"] += (
            " Mặc thoải mái suốt ngày dài."
        )
        with self.assertRaisesRegex(ValueError, "unsupported claim"):
            llm_explain._validate_generated(
                {int(entry["rank"]): entry for entry in payload["explanations"]},
                REQUEST,
                THREE_RESULT["outfits"],
            )

    def test_invalid_output_is_retried_once_then_falls_back(self):
        result = json.loads(json.dumps(THREE_RESULT))
        duplicate = _structured_payload()
        duplicate_entry = duplicate["explanations"][0].copy()
        duplicate_entry["rank"] = 2
        duplicate_entry["why_suitable"] = duplicate_entry["why_suitable"].replace(
            "Áo sơ mi trắng", "Áo len mỏng"
        ).replace("8.1", "37")
        duplicate["explanations"][1] = duplicate_entry
        client, context_client = _client_mock([duplicate, duplicate])
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("api.llm_explain.genai.Client", return_value=context_client),
        ):
            llm_explain.add_explanations(result, REQUEST)
        self.assertEqual(client.models.generate_content.call_count, 2)
        self.assertTrue(all(item["explanation_source"] == "rules" for item in result["outfits"]))
        self.assertTrue(
            all("kiểm tra dữ kiện" in item["explanation_notice"] for item in result["outfits"])
        )

    def test_gemini_service_error_or_quota_uses_fallback_without_leaking_error(self):
        for error in (RuntimeError("service unavailable"), RuntimeError("quota exhausted")):
            result = json.loads(json.dumps(THREE_RESULT))
            context_client = Mock()
            context_client.__enter__ = Mock(side_effect=error)
            context_client.__exit__ = Mock(return_value=False)
            with (
                patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
                patch("api.llm_explain.genai.Client", return_value=context_client),
            ):
                llm_explain.add_explanations(result, REQUEST)
            self.assertTrue(all(item["explanation_source"] == "rules" for item in result["outfits"]))
            self.assertTrue(all(item["explanation_llm"] for item in result["outfits"]))
            self.assertEqual(
                len({item["explanation_llm"] for item in result["outfits"]}), 3
            )
            self.assertTrue(
                all(
                    "điểm model" in item["explanation_structured"]["why_suitable"].casefold()
                    and "xếp hạng" in item["explanation_structured"]["why_suitable"].casefold()
                    and "nhiệt độ" in item["explanation_structured"]["weather_fit"].casefold()
                    and "Khoảng điểm model" in item["explanation_structured"]["caution"]
                    for item in result["outfits"]
                )
            )

    def test_malformed_json_falls_back(self):
        result = json.loads(json.dumps(THREE_RESULT))
        client, context_client = _client_mock([{}])
        client.models.generate_content.side_effect = [Mock(text="{bad json"), Mock(text="{still bad")]
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}),
            patch("api.llm_explain.genai.Client", return_value=context_client),
        ):
            llm_explain.add_explanations(result, REQUEST)
        self.assertEqual(client.models.generate_content.call_count, 2)
        self.assertTrue(all(item["explanation_source"] == "rules" for item in result["outfits"]))

    def test_missing_model_configuration_skips_gemini(self):
        result = json.loads(json.dumps(THREE_RESULT))
        with (
            patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": ""}),
            patch("api.llm_explain.genai.Client") as client_factory,
        ):
            llm_explain.add_explanations(result, REQUEST)
        client_factory.assert_not_called()
        self.assertTrue(all(item["explanation_source"] == "rules" for item in result["outfits"]))
        self.assertTrue(all("GEMINI_MODEL" in item["explanation_notice"] for item in result["outfits"]))

    def test_private_free_text_is_excluded_from_evidence_and_cache_key(self):
        request_with_private_data = {
            **REQUEST,
            "notes": "PRIVATE_NOTE_DO_NOT_SEND",
            "schedule_summary": "PRIVATE_SCHEDULE_DO_NOT_SEND",
            "gender": "PRIVATE_GENDER_DO_NOT_SEND",
        }
        evidence = json.dumps(
            llm_explain._evidence(request_with_private_data, THREE_RESULT["outfits"]),
            ensure_ascii=False,
        )
        self.assertNotIn("PRIVATE_", evidence)
        self.assertEqual(
            llm_explain._cache_key("test-model", REQUEST, THREE_RESULT["outfits"]),
            llm_explain._cache_key(
                "test-model", request_with_private_data, THREE_RESULT["outfits"]
            ),
        )

    def test_rule_explanations_are_distinct_across_five_weather_contexts(self):
        contexts = [
            (12, 0, "Đi học"),
            (18, 20, "Đi học"),
            (24, 90, "Đi học"),
            (30, 0, "Xưởng/Công trình"),
            (16, 65, "Vẽ ngoại cảnh"),
        ]
        for temperature, rain_probability, activity in contexts:
            with self.subTest(temperature=temperature, rain=rain_probability, activity=activity):
                result = json.loads(json.dumps(THREE_RESULT))
                request = {
                    **REQUEST,
                    "temperature": temperature,
                    "rain_probability": rain_probability,
                    "activity": activity,
                }
                llm_explain.add_explanations(result, request)
                reasons = [
                    outfit["explanation_structured"]["why_suitable"]
                    for outfit in result["outfits"]
                ]
                self.assertEqual(len(set(reasons)), 3)
                self.assertTrue(all("**" in reason for reason in reasons))
                self.assertTrue(
                    all(f"{temperature:g}°C" in outfit["explanation_structured"]["weather_fit"]
                        for outfit in result["outfits"])
                )


if __name__ == "__main__":
    unittest.main()
