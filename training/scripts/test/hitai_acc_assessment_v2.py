import os
from os import makedirs as mkdirs
from os.path import join
import pandas as pd
import numpy as np
from typing import Dict


class HitAIAccuracyAssessment:
    """
    Accuracy, Precision, Recall assessment for
    HitAI Cloud & Shadow Detection using LOCAL CSV Ground Truth
    """

    # --------------------------------------------------
    # INIT
    # --------------------------------------------------
    def __init__(self, hitai_csv_path: str, gt_csv_path: str):
        self.hitai_csv_path = hitai_csv_path
        self.gt_csv_path = gt_csv_path
        self.df = None
        self.metrics: Dict[str, Dict] = {}

    # --------------------------------------------------
    # UTILITIES
    # --------------------------------------------------
    @staticmethod
    def _normalize_filename(name: str) -> str:
        if pd.isna(name):
            return ""
        name = str(name).strip()
        base, ext = os.path.splitext(name)
        return base if ext else name

    @staticmethod
    def _to_int(val) -> int:
        if isinstance(val, str):
            return 1 if val.strip().upper() in ["TRUE", "1", "YES"] else 0
        return int(bool(val))

    # --------------------------------------------------
    # LOAD HITAI RESULTS
    # --------------------------------------------------
    def load_hitai_results(self) -> pd.DataFrame:
        print(f"Loading HitAI CSV: {self.hitai_csv_path}")

        df = pd.read_csv(self.hitai_csv_path)
        df.columns = [c.lower().strip() for c in df.columns]

        required = ["name", "cloud", "cloud_shadow"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column in HitAI CSV: {col}")

        df["name"] = df["name"].apply(self._normalize_filename)
        df["pred_cloud"] = df["cloud"].apply(self._to_int)
        df["pred_shadow"] = df["cloud_shadow"].apply(self._to_int)

        return df[["name", "pred_cloud", "pred_shadow"]]

    # --------------------------------------------------
    # LOAD GROUND TRUTH (LOCAL CSV)
    # --------------------------------------------------
    def load_ground_truth(self) -> pd.DataFrame:
        print(f"Loading Ground Truth CSV: {self.gt_csv_path}")

        gt_df = pd.read_csv(self.gt_csv_path)
        gt_df.columns = [c.lower().strip() for c in gt_df.columns]

        required = ["image_name", "cloud", "shadow"]
        for col in required:
            if col not in gt_df.columns:
                raise ValueError(f"Missing column in GT CSV: {col}")

        gt_df["name"] = gt_df["image_name"].apply(self._normalize_filename)
        gt_df["gt_cloud"] = gt_df["cloud"].apply(self._to_int)
        gt_df["gt_shadow"] = gt_df["shadow"].apply(self._to_int)

        return gt_df[["name", "gt_cloud", "gt_shadow"]]

    # --------------------------------------------------
    # PREPARE MERGED DATAFRAME
    # --------------------------------------------------
    def prepare_dataframe(self) -> pd.DataFrame:
        hitai_df = self.load_hitai_results()
        gt_df = self.load_ground_truth()

        # OUTER MERGE = union of GT & HitAI
        df = pd.merge(
            gt_df,
            hitai_df,
            on="name",
            how="outer"
        ).fillna(0)

        # Convert to int
        for col in ["gt_cloud", "gt_shadow", "pred_cloud", "pred_shadow"]:
            df[col] = df[col].astype(int)

        # 🚫 Drop pure negatives
        df = df[~(
            (df.gt_cloud == 0) &
            (df.gt_shadow == 0) &
            (df.pred_cloud == 0) &
            (df.pred_shadow == 0)
        )]

        self.df = df.reset_index(drop=True)

        print(f"Prepared dataframe with {len(self.df)} samples")

        return self.df

    # --------------------------------------------------
    # METRICS
    # --------------------------------------------------
    def calculate_class_metrics(self, gt_col: str, pred_col: str) -> Dict:
        y_true = self.df[gt_col]
        y_pred = self.df[pred_col]

        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()

        detection_accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

        return {
            "Total Samples": int(tp + tn + fp + fn),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Detection Accuracy": detection_accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

    # --------------------------------------------------
    # RUN ASSESSMENT
    # --------------------------------------------------
    def run_assessment(self):
        self.metrics["Cloud"] = self.calculate_class_metrics("gt_cloud", "pred_cloud")
        self.metrics["Shadow"] = self.calculate_class_metrics("gt_shadow", "pred_shadow")

        self.df["gt_combined"] = self.df[["gt_cloud", "gt_shadow"]].max(axis=1)
        self.df["pred_combined"] = self.df[["pred_cloud", "pred_shadow"]].max(axis=1)

        self.metrics["Cloud+Shadow"] = self.calculate_class_metrics(
            "gt_combined", "pred_combined"
        )

    # --------------------------------------------------
    # REPORT
    # --------------------------------------------------
    def print_report(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("        HitAI ACCURACY ASSESSMENT REPORT        ")
        lines.append("=" * 60)

        for k, v in self.metrics.items():
            lines.append(f"\n--- {k} Detection ---")
            for m, val in v.items():
                if isinstance(val, float):
                    lines.append(f"{m:<22}: {val:.4f}")
                else:
                    lines.append(f"{m:<22}: {val}")

        report = "\n".join(lines)
        print(report)
        return report

    # --------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------
    def save_results(self, out_csv: str, out_txt: str, report: str):
        self.df.to_csv(out_csv, index=False)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(report)

    # --------------------------------------------------
    # CONFUSION LISTS
    # --------------------------------------------------
    def get_detection_lists_df(self) -> pd.DataFrame:
        data = {}

        for cat in ["cloud", "shadow"]:
            gt = f"gt_{cat}"
            pred = f"pred_{cat}"
            Cat = cat.capitalize()

            data[f"{Cat}_TP"] = self.df[(self.df[gt]==1) & (self.df[pred]==1)]["name"].tolist()
            data[f"{Cat}_FP"] = self.df[(self.df[gt]==0) & (self.df[pred]==1)]["name"].tolist()
            data[f"{Cat}_FN"] = self.df[(self.df[gt]==1) & (self.df[pred]==0)]["name"].tolist()
            data[f"{Cat}_TN"] = self.df[(self.df[gt]==0) & (self.df[pred]==0)]["name"].tolist()

        max_len = max(len(v) for v in data.values())
        for k in data:
            data[k].extend([np.nan] * (max_len - len(data[k])))

        return pd.DataFrame(data)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    HITAI_RESULT_PATH = r"D:\projects\Image_QC_GUI\2_Repos\OmniCloudMask\training\data\qc_results\custom_model\1-30-26\1\quality_control_results.csv"

    GT_CSV_PATH = r"E:\ImageQC\dataset\Test_all_gt\gt_test.csv"

    VERSION = 1.4
    OUT_DIR = join(os.path.dirname(HITAI_RESULT_PATH), "hitai_assessment_results")
    mkdirs(OUT_DIR, exist_ok=True)

    assessor = HitAIAccuracyAssessment(HITAI_RESULT_PATH, GT_CSV_PATH)

    assessor.prepare_dataframe()
    assessor.run_assessment()
    report = assessor.print_report()

    assessor.save_results(
        join(OUT_DIR, f"hitai_vs_gt_comparison_{VERSION}.csv"),
        join(OUT_DIR, f"hitai_vs_gt_report_{VERSION}.txt"),
        report
    )

    df_lists = assessor.get_detection_lists_df()
    df_lists.to_csv(join(OUT_DIR, f"hitai_vs_gt_confusion_lists_{VERSION}.csv"), index=False)

    print("\n✅ Assessment completed successfully.")