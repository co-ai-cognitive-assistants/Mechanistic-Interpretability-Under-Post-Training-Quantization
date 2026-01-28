import os

MODELS_DIR = "models"
METHODS = ["float16", "bfloat16", "int8", "int4", "gguf", "awq", "gptq","hqq","fp8"]
CHECK = "✅"
CROSS = "❌"
GREEN_COLOR = "\033[92m"
RESET_COLOR = "\033[0m"

def main():
    if not os.path.exists(MODELS_DIR):
        print(f"Directory '{MODELS_DIR}' not found.")
        return

    # Get list of model directories
    models = sorted([d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d)) and "gpt-oss" not in d])
    
    if not models:
        print("No models found.")
        return

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
        statuses = []
        for method in METHODS:
            path = os.path.join(MODELS_DIR, model, method)
            # Check if directory exists and is not empty
            exists = os.path.isdir(path) and len(os.listdir(path)) > 0
            statuses.append(exists)
        
        all_complete = all(statuses)
        
        # Prefix and formatting
        prefix = CHECK if all_complete else CROSS
        
        if all_complete:
            # Green name for fully completed models
            model_display = f"{prefix} {GREEN_COLOR}{model}{RESET_COLOR}"
        else:
            model_display = f"{prefix} {model}"
            
        # Calculate padding manually to handle ANSI codes correctly
        # "Visual" length roughly equals 2 (emoji + space) + len(model)
        # Note: Emojis often take 2 visual columns but len()=1. 
        # Aligning strictly by char count usually works if emojis are consistent.
        visible_len = 2 + len(model)
        padding = model_col_width - visible_len
        if padding < 0: padding = 0
        
        row = f"{model_display}{' ' * padding}"

        for i, method in enumerate(METHODS):
            exists = statuses[i]
            
            # Center the mark
            # Emojis take 2 visual spaces, so subtract 2 from width to calculate padding
            padding_total = method_cols_width[method] - 2
            if padding_total < 0: padding_total = 0
            pad_l = padding_total // 2
            pad_r = padding_total - pad_l
            
            mark = CHECK if exists else CROSS
            cell = " " * pad_l + mark + " " * pad_r
            row += f"|{cell}"
        print(row)
    print(separator)

if __name__ == "__main__":
    main()