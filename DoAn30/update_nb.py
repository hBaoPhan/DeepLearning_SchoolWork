import json

file_path = 'Đồ_Án_30_DeepLearning (4).ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        src_str = "".join(source)
        if "ax2.plot(y_test_plot, label='Dữ liệu thực tế" in src_str:
            new_source = []
            for line in source:
                new_source.append(line)
                if line.startswith("y_test_plot = y_test_seq[:, 0, 0]"):
                    new_source.append("pred_test_plot = pred_test_seq[:, 0, 0]\n")
                if "ax2.plot(y_test_plot, label='Dữ liệu thực tế" in line:
                    new_source.append("ax2.plot(pred_test_plot, label='Dự báo AI', color='magenta', linestyle='dashed', alpha=0.8)\n")
            cell['source'] = new_source

        if "ax2.plot(y_new_plot, label='Dữ liệu thực tế" in src_str:
            new_source = []
            for line in source:
                new_source.append(line)
                if line.startswith("y_new_plot = y_new_seq[:, 0, 0]"):
                    new_source.append("pred_new_plot = pred_new_seq[:, 0, 0]\n")
                if "ax2.plot(y_new_plot, label='Dữ liệu thực tế" in line:
                    new_source.append("ax2.plot(pred_new_plot, label='Dự báo AI', color='magenta', linestyle='dashed', alpha=0.8)\n")
            cell['source'] = new_source

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
