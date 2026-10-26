"""Recommendation coverage for temperature, score thresholds and MMR diversity."""

import unittest
from unittest.mock import patch

import pandas as pd
from pydantic import ValidationError

from fastapi.testclient import TestClient

from api import main
from api.main import CATALOG, MODEL_ARTIFACT
from api import recommender
from api.config import DATASET_COLORS, OUTFIT_MIN_SCORE, OUTFIT_MMR_LAMBDA, OUTFIT_SCORE_BAND
from api.recommender import _content_similarity, _dataset_color_match, _score_catalog, recommend
from api.schemas import RecommendRequest


class RecommenderTests(unittest.TestCase):
    def test_selector_memoization_preserves_original_ranked_combinations(self):
        shirts = [
            {"ID_SP": f"s{index}", "score": score, "rank_score": rank,
             "Phong cách": style, "Màu sắc": color}
            for index, (score, rank, style, color) in enumerate([
                (9.1, 8.8, "Casual", "Trắng"), (8.8, 8.9, "Minimalist", "Đen"),
                (8.3, 8.2, "Sporty", "Xanh"), (7.9, 8.4, "Artsy", "Be"),
            ])
        ]
        lowers = [
            {"ID_SP": f"p{index}", "score": score, "rank_score": rank,
             "Phong cách": style, "Màu sắc": color}
            for index, (score, rank, style, color) in enumerate([
                (8.9, 8.5, "Casual", "Đen"), (8.6, 8.6, "Minimalist", "Xám"),
                (8.2, 8.0, "Sporty", "Navy"), (7.8, 8.3, "Artsy", "Nâu"),
            ])
        ]
        shoes = [
            {"ID_SP": f"g{index}", "score": score, "rank_score": rank,
             "Phong cách": style, "Màu sắc": color}
            for index, (score, rank, style, color) in enumerate([
                (8.7, 8.6, "Casual", "Trắng"), (8.5, 8.7, "Minimalist", "Đen"),
                (8.0, 7.9, "Sporty", "Xanh"), (7.7, 8.1, "Artsy", "Nâu"),
            ])
        ]
        combos = [[shirt, lower, shoe] for shirt in shirts for lower in lowers for shoe in shoes]

        def original_selector(candidates, count):
            best_score = max(recommender._combo_model_score(combo) for combo in candidates)

            def choose(current):
                current.sort(key=lambda combo: (-recommender._combo_rank(combo),
                                                tuple(str(item["ID_SP"]) for item in combo)))
                chosen_items, remaining = [], list(current)
                while remaining and len(chosen_items) < count:
                    available = [combo for combo in remaining if all(
                        not (recommender._main_item_ids(combo) & recommender._main_item_ids(prior))
                        for prior in chosen_items
                    )]
                    if not available:
                        break
                    picked = max(available, key=lambda combo: (
                        (1 - OUTFIT_MMR_LAMBDA) * (recommender._combo_rank(combo) / 10)
                        - OUTFIT_MMR_LAMBDA * max(
                            (recommender._combo_similarity(combo, prior) for prior in chosen_items),
                            default=0.0,
                        ),
                        recommender._combo_rank(combo),
                    ))
                    chosen_items.append(picked)
                    remaining.remove(picked)
                return chosen_items

            maximum_band = max(0.0, best_score - OUTFIT_MIN_SCORE)
            band = min(OUTFIT_SCORE_BAND, maximum_band)
            while True:
                current = [combo for combo in candidates if recommender._combo_model_score(combo)
                           >= max(OUTFIT_MIN_SCORE, best_score - band)]
                selected = choose(current)
                if len(selected) >= count or band >= maximum_band:
                    break
                band = min(maximum_band, band + 0.25)
            return sorted(selected, key=recommender._combo_model_score, reverse=True)

        expected = [[item["ID_SP"] for item in combo] for combo in original_selector(combos, 3)]
        with (
            patch("api.recommender._combo_model_score", wraps=recommender._combo_model_score) as score_mock,
            patch("api.recommender._combo_rank", wraps=recommender._combo_rank) as rank_mock,
        ):
            actual = recommender._select_diverse(combos, 3)
        self.assertEqual([[item["ID_SP"] for item in combo] for combo in actual], expected)
        self.assertLessEqual(score_mock.call_count, len(combos))
        self.assertLessEqual(rank_mock.call_count, len(combos))

    def test_unisex_is_a_valid_request_and_top_n_cannot_exceed_three(self):
        request = RecommendRequest(
            gender="Unisex", activity="Đi học", temperature=22, top_n=3
        )
        self.assertEqual(request.gender, "Unisex")
        with self.assertRaises(ValidationError):
            RecommendRequest(
                gender="Unisex", activity="Đi học", temperature=22, top_n=4
            )

    def test_selected_color_is_a_hard_filter_and_aliases_are_explicit(self):
        self.assertTrue(_dataset_color_match("Đen trắng", ["Đen"]))
        self.assertFalse(_dataset_color_match("Đỏ", ["Đen"]))
        catalog = CATALOG.head(40).copy().reset_index(drop=True)
        buckets, _ = _score_catalog(
            catalog, MODEL_ARTIFACT, activity="Đi học", temperature=22,
            gender="Nam", styles=[], colors=["Đen"], formality=35,
        )
        self.assertTrue(all(
            item["Màu sắc"] in {"Đen", "Đen trắng"}
            for items in buckets.values() for item in items
        ))

    def test_every_metadata_color_is_valid_and_sparse_color_fills_complete_outfits(self):
        request = RecommendRequest(
            gender="Nữ", activity="Đi học", temperature=22,
            colors=list(DATASET_COLORS), top_n=3,
        )
        self.assertEqual(set(request.colors), set(DATASET_COLORS))
        self.assertTrue(_dataset_color_match("Đỏ đô", ["Đỏ đô"]))
        self.assertFalse(_dataset_color_match("Đỏ", ["Đỏ đô"]))

        sparse_request = RecommendRequest(
            gender="Nữ", activity="Đi học", temperature=22,
            colors=["Trong suốt"], top_n=3,
        )
        result = recommend(CATALOG, MODEL_ARTIFACT, sparse_request)
        self.assertEqual(len(result["outfits"]), 3)
        self.assertTrue(any("Không đủ combo hoàn chỉnh" in warning for warning in result["warnings"]))

    def test_binary_gender_filters_do_not_admit_unisex_items(self):
        for gender in ("Nam", "Nữ"):
            buckets, _ = _score_catalog(
                CATALOG, MODEL_ARTIFACT, activity="Đi học", temperature=22,
                gender=gender, styles=[], colors=[], formality=35,
            )
            self.assertTrue(all(
                item["Giới tính"] == gender
                for items in buckets.values() for item in items
            ))

    def test_duplicate_catalog_ids_are_not_recommended_twice(self):
        catalog = pd.concat([CATALOG, CATALOG.head(1)], ignore_index=True)
        buckets, _ = _score_catalog(
            catalog, MODEL_ARTIFACT, activity="Đi học", temperature=22,
            gender="Nam", styles=[], colors=[], formality=35,
        )
        ids = [item["ID_SP"] for items in buckets.values() for item in items]
        self.assertEqual(len(ids), len(set(ids)))

    def test_content_profile_uses_explicit_style_color_and_formality(self):
        catalog = CATALOG.head(30).copy().reset_index(drop=True)
        similarity = _content_similarity(
            catalog,
            styles=["Casual"],
            colors=["Đen"],
            formality=35,
        )
        self.assertEqual(len(similarity), len(catalog))
        self.assertTrue(((similarity >= 0) & (similarity <= 1)).all())
        self.assertTrue((similarity > 0).any())

    def test_predict_http_endpoint_returns_score_interval_and_confidence(self):
        response = TestClient(main.app).post(
            "/predict",
            json={
                "Hoàn cảnh": "Đi học",
                "Phong cách": "Casual",
                "Tmin (°C)": 18,
                "Tmax (°C)": 30,
                "Nhiệt độ ngày (°C)": 24,
                "Giữ ấm": 3,
                "Thoáng khí": 4,
                "Vận động": 4,
                "Chỉn chu": 3,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            {
                "score",
                "score_low",
                "score_high",
                "std_between_trees",
                "confidence_level",
                "model_version",
            }
            <= response.json().keys()
        )

    def test_twenty_contexts_return_high_scoring_pairwise_distinct_outfits(self):
        cases = [
            (gender, activity, temperature, "Casual")
            for gender in ("Nam", "Nữ")
            for activity in (
                "Đi học",
                "Xưởng/Công trình",
                "Vẽ ngoại cảnh",
                "Thuyết trình",
            )
            for temperature in (12, 28)
        ]
        cases.extend(
            [
                ("Nam", "Đi học", 20, "Smart Casual"),
                ("Nữ", "Xưởng/Công trình", 34, "Streetwear"),
                ("Nam", "Vẽ ngoại cảnh", 34, "Minimalist"),
                ("Nữ", "Thuyết trình", 12, "Artsy"),
            ]
        )
        self.assertEqual(len(cases), 20)
        for gender, activity, temperature, style in cases:
            with self.subTest(gender=gender, activity=activity, temperature=temperature, style=style):
                result = recommend(
                    CATALOG,
                    MODEL_ARTIFACT,
                    RecommendRequest(
                        gender=gender,
                        activity=activity,
                        temperature=temperature,
                        styles=[style],
                        top_n=3,
                    ),
                )
                outfits = result["outfits"]
                self.assertLessEqual(len(outfits), 3)
                if len(outfits) < 3:
                    self.assertTrue(any("Chỉ ghép được" in warning for warning in result["warnings"]))
                if not outfits:
                    continue
                self.assertTrue(all(item["score"] >= 6.0 for item in outfits))
                self.assertTrue(
                    all(0 <= piece["content_match"] <= 100
                        for outfit in outfits for piece in outfit["items"])
                )
                self.assertEqual(
                    [item["score"] for item in outfits],
                    sorted((item["score"] for item in outfits), reverse=True),
                )
                for index, outfit in enumerate(outfits):
                    self.assertIn("explanation_summary", outfit)
                    self.assertIn(activity, outfit["explanation_summary"])
                    self.assertIn(f"{temperature:g}°C", outfit["explanation_summary"])
                    main_items = {
                        item["id"]
                        for item in outfit["items"]
                        if item["category"] not in {"Giày", "Dép"}
                    }
                    for other in outfits[index + 1:]:
                        other_main = {
                            item["id"]
                            for item in other["items"]
                            if item["category"] not in {"Giày", "Dép"}
                        }
                        self.assertFalse(main_items & other_main)
                if temperature == 12:
                    self.assertTrue(
                        all(
                            any(item["category"] == "Áo khoác" for item in outfit["items"])
                            for outfit in outfits
                        )
                    )
                if temperature == 34:
                    self.assertTrue(
                        all(
                            all(item["category"] != "Áo khoác" for item in outfit["items"])
                            for outfit in outfits
                        )
                    )


if __name__ == "__main__":
    unittest.main()
