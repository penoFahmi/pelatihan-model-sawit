import json

def fix_notebook():
    with open('pelatihan_mobilenetv2_v2.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            
            # Uncomment Phase 1
            if 'history_phase1 = model.fit(' in source and '#' in source:
                source = source.replace('# history_phase1 = model.fit(', 'history_phase1 = model.fit(')
                source = source.replace('#     train_gen,', '    train_gen,')
                source = source.replace('#     validation_data=val_gen,', '    validation_data=val_gen,')
                source = source.replace('#     epochs=20,', '    epochs=20,')
                source = source.replace('#     class_weight=class_weight_dict,', '    class_weight=class_weight_dict,')
                source = source.replace('#     callbacks=callbacks_list', '    callbacks=callbacks_list')
                source = source.replace('# )', ')')
                
                # Change learning rate to 1e-3 for faster top-layer convergence
                source = source.replace('learning_rate=1e-4', 'learning_rate=1e-3')
                
                cell['source'] = [line + '\n' for line in source.split('\n')]
                if cell['source'][-1] == '\n':
                    cell['source'] = cell['source'][:-1]
                else:
                    cell['source'][-1] = cell['source'][-1][:-1]

            # Uncomment Phase 2
            if 'history_phase2 = model.fit(' in source and '#' in source:
                source = source.replace('# history_phase2 = model.fit(', 'history_phase2 = model.fit(')
                source = source.replace('#     train_gen,', '    train_gen,')
                source = source.replace('#     validation_data=val_gen,', '    validation_data=val_gen,')
                source = source.replace('#     epochs=30,', '    epochs=30,')
                source = source.replace('#     class_weight=class_weight_dict,', '    class_weight=class_weight_dict,')
                source = source.replace('#     callbacks=callbacks_list', '    callbacks=callbacks_list')
                source = source.replace('# )', ')')
                
                cell['source'] = [line + '\n' for line in source.split('\n')]
                if cell['source'][-1] == '\n':
                    cell['source'] = cell['source'][:-1]
                else:
                    cell['source'][-1] = cell['source'][-1][:-1]
                    
            # Uncomment TFLite conversion
            if 'tflite_quant_model = converter.convert()' in source and '#' in source:
                source = source.replace('# tflite_quant_model = converter.convert()', 'tflite_quant_model = converter.convert()')
                source = source.replace('# tflite_filepath = "mobilenetv2_sawit_fp16.tflite"', 'tflite_filepath = "mobilenetv2_sawit_fp16.tflite"')
                source = source.replace('# with open(tflite_filepath, "wb") as f:', 'with open(tflite_filepath, "wb") as f:')
                source = source.replace('#     f.write(tflite_quant_model)', '    f.write(tflite_quant_model)')
                source = source.replace('# print(f"Model TFLite tersimpan sebagai {tflite_filepath}")', 'print(f"Model TFLite tersimpan sebagai {tflite_filepath}")')
                
                cell['source'] = [line + '\n' for line in source.split('\n')]
                if cell['source'][-1] == '\n':
                    cell['source'] = cell['source'][:-1]
                else:
                    cell['source'][-1] = cell['source'][-1][:-1]

    with open('pelatihan_mobilenetv2_v2.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

if __name__ == '__main__':
    fix_notebook()
    print("Fixed!")
