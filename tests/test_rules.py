import unittest
from unittest.mock import patch

from qwen_text_eval.rules import MEDICAL_DISCLAIMER, count_characters, evaluate_rules


def checks(result):
    return {check["id"]: check for check in result["checks"]}


def interpretation(overview="一张餐厅菜单，列出菜品和价格。", points=2, modules=1):
    body = [overview, ""]
    for index in range(modules):
        body.extend([f"### 【模块{index + 1}】", *[f"- 菜品{point + 1}，10元。" for point in range(points)], ""])
    return "\n".join(body)


class RulesTests(unittest.TestCase):
    def test_counting_includes_punctuation_digits_and_foreign_characters(self):
        self.assertEqual(count_characters("菜 12，A\n\t"), 5)
        self.assertEqual(count_characters("𠮷"), 1)

    def test_summary_exact_boundaries_and_no_mutation(self):
        text = "菜" * 10 + "\n" + "价" * 29 + "。"
        result = evaluate_rules(text, "summary")
        by_id = checks(result)
        self.assertEqual(result["char_count"], 40)
        self.assertEqual(by_id["summary_title_length"]["status"], "pass")
        self.assertEqual(by_id["summary_description_length"]["status"], "pass")
        self.assertEqual(text, "菜" * 10 + "\n" + "价" * 29 + "。")

    def test_summary_over_boundaries(self):
        result = evaluate_rules("菜" * 11 + "\n" + "价" * 31, "summary")
        self.assertEqual(checks(result)["summary_title_length"]["status"], "fail")
        self.assertEqual(checks(result)["summary_description_length"]["status"], "fail")
        self.assertFalse(result["hard_pass"])

    def test_crlf_and_one_terminal_newline_are_allowed(self):
        result = evaluate_rules("菜单\r\n含米饭与汤，价格为12元。\r\n", "summary")
        self.assertEqual(checks(result)["summary_line_count"]["status"], "pass")
        self.assertEqual(checks(result)["summary_no_blank_lines"]["status"], "pass")

    def test_multiple_terminal_newlines_are_rejected(self):
        result = evaluate_rules("菜单\n包含米饭。\n\n", "summary")
        self.assertEqual(checks(result)["summary_line_count"]["status"], "fail")
        self.assertEqual(checks(result)["summary_no_blank_lines"]["status"], "fail")

    def test_blank_or_third_line_is_not_silently_removed(self):
        for text in ("菜单\n \n价格12元。", "菜单\n价格12元。\n还有汤。", "\n菜单\n价格12元。"):
            with self.subTest(text=text):
                result = evaluate_rules(text, "summary")
                self.assertEqual(checks(result)["summary_line_count"]["status"], "fail")
                self.assertFalse(result["hard_pass"])

    def test_empty_output_is_failure(self):
        for task in ("summary", "interpretation"):
            for text in ("", " \n\t"):
                result = evaluate_rules(text, task)
                self.assertFalse(result["hard_pass"])
                self.assertFalse(result["needs_review"])

    def test_summary_has_no_markdown_or_forbidden_prefix(self):
        for text in ("**菜单**\n包含米饭。", "标题：菜单\n描述：包含米饭。", "```\n菜单\n包含米饭。\n```"):
            result = evaluate_rules(text, "summary")
            self.assertFalse(result["hard_pass"])

    def test_interpretation_markdown_is_allowed(self):
        result = evaluate_rules(interpretation(), "interpretation", "menu")
        self.assertNotIn("summary_markdown", checks(result))
        self.assertEqual(checks(result)["interpretation_module_count"]["status"], "pass")
        self.assertEqual(checks(result)["interpretation_module_points"]["status"], "pass")

    def test_thinking_tags_are_rejected_in_both_tasks(self):
        for task in ("summary", "interpretation"):
            result = evaluate_rules("<think>检查图片</think>\n菜单\n包含米饭。", task)
            self.assertEqual(checks(result)["thinking_markup"]["status"], "fail")

    def test_emoji_detection_does_not_confuse_chinese_or_circled_numerals(self):
        for text in ("菜单一二三\n①汤②米饭，12元。", "花王®眼罩\n产品规格12片。"):
            self.assertEqual(checks(evaluate_rules(text, "summary"))["emoji"]["status"], "pass")
        for text in ("菜单🍜\n含米饭。", "菜单\n含米饭✅。", "菜单\n1️⃣米饭。", "菜单\n日本🇯🇵。"):
            self.assertEqual(checks(evaluate_rules(text, "summary"))["emoji"]["status"], "fail")

    def test_foreign_brands_are_reviewed_never_blanket_failed(self):
        result = evaluate_rules(interpretation("Swisse保健品，规格30粒。"), "interpretation", "medicine")
        self.assertEqual(checks(result)["chinese_language"]["status"], "review")
        self.assertTrue(result["needs_review"])

    def test_foreign_parentheses_are_reviewed(self):
        result = evaluate_rules("菜单\n提供米饭（rice）。", "summary")
        self.assertEqual(checks(result)["foreign_parenthetical"]["status"], "review")
        self.assertEqual(checks(evaluate_rules("菜单\n提供米饭（配汤）。", "summary"))["foreign_parenthetical"]["status"], "pass")

    def test_opencc_candidate_is_flagged_without_changing_output(self):
        class Converter:
            def convert(self, text):
                return text.replace("廳", "厅")

        text = "餐廳菜单\n供应米饭。"
        with patch("qwen_text_eval.rules._opencc_converter", return_value=Converter()):
            result = evaluate_rules(text, "summary")
        self.assertEqual(checks(result)["simplified_chinese"]["status"], "review")
        self.assertEqual(checks(result)["simplified_chinese"]["observed"]["candidate_characters"], ["廳"])
        self.assertIn("廳", text)

    def test_missing_opencc_reports_review(self):
        with patch("qwen_text_eval.rules._opencc_converter", return_value=None):
            result = evaluate_rules("餐厅菜单\n含米饭。", "summary")
        self.assertEqual(checks(result)["simplified_chinese"]["status"], "review")

    def test_subjective_wording_is_reviewed_in_context(self):
        result = evaluate_rules("菜单\n招牌上写着超值套餐。", "summary")
        self.assertEqual(checks(result)["subjective_wording"]["status"], "review")

    def test_overview_length_boundary(self):
        for count, expected in ((50, "pass"), (51, "fail")):
            result = evaluate_rules(interpretation("菜" * count), "interpretation", "menu")
            self.assertEqual(checks(result)["interpretation_overview"]["status"], expected)

    def test_missing_overview_is_failed(self):
        result = evaluate_rules("【信息】\n- 米饭\n- 汤", "interpretation", "menu")
        self.assertEqual(checks(result)["interpretation_overview"]["status"], "fail")

    def test_module_count_and_bullet_boundaries(self):
        for module_count, expected in ((0, "fail"), (1, "pass"), (5, "pass"), (6, "fail")):
            result = evaluate_rules(interpretation(modules=module_count), "interpretation", "menu")
            self.assertEqual(checks(result)["interpretation_module_count"]["status"], expected)
        for points, expected in ((1, "fail"), (2, "pass"), (4, "pass"), (5, "fail")):
            result = evaluate_rules(interpretation(points=points), "interpretation", "menu")
            self.assertEqual(checks(result)["interpretation_module_points"]["status"], expected)

    def test_numbered_points_and_bold_headers(self):
        result = evaluate_rules("菜单展示米饭和汤。\n\n**【菜品】**\n1. 米饭\n2、汤", "interpretation", "menu")
        self.assertEqual(checks(result)["interpretation_module_points"]["status"], "pass")

    def test_length_limit_uses_reference_scene_and_menu_only_exception(self):
        text = interpretation() + "菜" * 501
        for scene, expected in (("menu", "not_applicable"), ("餐厅菜单", "not_applicable"),
                                ("restaurant", "fail"), ("general", "fail"), (None, "review")):
            result = evaluate_rules(text, "interpretation", scene)
            self.assertEqual(checks(result)["interpretation_total_length"]["status"], expected)

    def test_short_output_does_not_need_menu_exception(self):
        result = evaluate_rules(interpretation(), "interpretation")
        self.assertEqual(checks(result)["interpretation_total_length"]["status"], "pass")
        self.assertEqual(checks(result)["medical_disclaimer"]["status"], "review")

    def test_medical_disclaimer_required_at_end_only_for_known_medicine(self):
        text = interpretation()
        for scene in ("medicine", "药品保健品"):
            self.assertEqual(checks(evaluate_rules(text, "interpretation", scene))["medical_disclaimer"]["status"], "fail")
            result = evaluate_rules(text + MEDICAL_DISCLAIMER + "。 \n", "interpretation", scene)
            self.assertEqual(checks(result)["medical_disclaimer"]["status"], "pass")
            result = evaluate_rules(text + MEDICAL_DISCLAIMER + "\n还有建议。", "interpretation", scene)
            self.assertEqual(checks(result)["medical_disclaimer"]["status"], "fail")
        self.assertEqual(checks(evaluate_rules(text, "interpretation", "menu"))["medical_disclaimer"]["status"], "not_applicable")

    def test_tristate_review_does_not_become_success_or_zero_accuracy(self):
        result = evaluate_rules("菜单\n供应米饭和汤。", "summary")
        self.assertIsNone(result["hard_pass"])
        self.assertTrue(result["needs_review"])

    def test_input_validation(self):
        with self.assertRaises(TypeError):
            evaluate_rules(None, "summary")
        with self.assertRaises(ValueError):
            evaluate_rules("正文", "unknown")
        with self.assertRaises(TypeError):
            evaluate_rules("正文", "summary", 123)


if __name__ == "__main__":
    unittest.main()
