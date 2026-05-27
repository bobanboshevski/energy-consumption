export interface FeatureImportance {
    energy_demand: number;
    temp_max: number;
    temp_min: number;
    daylight_duration: number;
}

export interface ShapExplanation {
    date: string;
    predicted_demand: number;
    base_value: number;
    feature_importance: FeatureImportance;
    /** length = window_size (30). Index 0 = oldest day, index 29 = yesterday. */
    timestep_importance: number[];
    /** shape (window_size, n_features). Rows = timesteps, columns = features. */
    shap_matrix: number[][];
}

export interface ShapArtifact {
    generated_at: string;
    model_variant: string;
    shap_method: string;
    n_background_samples: number;
    feature_names: string[];
    window_size: number;
    n_explanations: number;
    version: string;
    explanations: ShapExplanation[];
}

export type ShapVariant = "keras" | "onnx" | "onnx_quantized";

export interface ExplainabilityErrors {
    keras: string | null;
    onnx: string | null;
    onnx_quantized: string | null;
}