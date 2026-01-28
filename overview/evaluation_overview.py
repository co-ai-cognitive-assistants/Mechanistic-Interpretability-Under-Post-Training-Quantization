import os

MODELS_DIR = "models"
RESULTS_DIR = "results"
METHODS = ["float16", "bfloat16", "int8", "int4", "gguf", "awq", "gptq", "hqq", "fp8"]
CHECK = "✅"
CROSS = "❌"
SKIP = "➖"
GREEN_COLOR = "\033[92m"
RESET_COLOR = "\033[0m"

def main():
    if not os.path.exists(MODELS_DIR):
        print(f"Directory '{MODELS_DIR}' not found.")
        return

    # Get list of model directories
    models = sorted([d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))])
    
    if not models:
        print("No models found.")
        return

    # Cache results files to avoid repeated directory listing
    # Results file format: {model}_{method}_temp_detail_{timestamp}.json
    results_files = []
    if os.path.exists(RESULTS_DIR):
        results_files = os.listdir(RESULTS_DIR)

    # Calculate column widths
    max_name_len = max((len(m) for m in models), default=10)
    # Width = max name + prefix (2 chars) + buffer
    model_col_width = max_name_len + 5
    
    method_cols_width = {m: max(len(m), 5) + 2 for m in METHODS}
    
    # Header
    header = f"{'Model Name':<{model_col_width}}"
    for m in METHODS:
        header += f"|{m:^{method_cols_width[m]}}"
    
    separator = "-" * len(header)
    
    print(separator)
    print(header)
    print(separator)

    for model in models:
        row_marks = []
        has_failures = False
        has_any_quant = False

        for method in METHODS:
            # Check if quantization folder exists (Baseline)
            quant_path = os.path.join(MODELS_DIR, model, method)
            quant_exists = os.path.isdir(quant_path)
            
            if quant_exists:
                has_any_quant = True
                # Check for matching result file
                # Construct expected prefix: model + "_" + method + "_temp_detail"
                prefix = f"{model}_{method}_temp_detail"
                
                # Check if any file in results starts with this prefix and ends with .json
                eval_exists = any(f.startswith(prefix) and f.endswith(".json") for f in results_files)
                
                if eval_exists:
                    row_marks.append(CHECK)
                else:
                    row_marks.append(CROSS)
                    has_failures = True
            else:
                # Quantization doesn't exist, so evaluation is not expected
                row_marks.append(SKIP)
        
        # Determine overall model status
        if has_any_quant and not has_failures:
            prefix = CHECK
            model_display = f"{prefix} {GREEN_COLOR}{model}{RESET_COLOR}"
        else:
            prefix = CROSS
            model_display = f"{prefix} {model}"
            
        # Calculate padding manually
        visible_len = 2 + len(model)
        padding = model_col_width - visible_len
        if padding < 0: padding = 0
        
        row = f"{model_display}{' ' * padding}"

        for i, method in enumerate(METHODS):
            mark = row_marks[i]
            
            # Center the mark
            padding_total = method_cols_width[method] - 2
            if padding_total < 0: padding_total = 0
            pad_l = padding_total // 2
            pad_r = padding_total - pad_l
            
            cell = " " * pad_l + mark + " " * pad_r
            row += f"|{cell}"
        print(row)
    print(separator)

if __name__ == "__main__":
    main()
