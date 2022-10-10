import os
import pickle
import pandas as pd
import numpy as np

def preprocess_mask_data(values_list, mode):
    values_matrix = np.array(values_list)
    input_size = values_matrix.shape[2]
    output_size = values_matrix.shape[1]
    labels = []
    changes = []
    
    for i in range(input_size):
        labels.append([])
        for j in range(output_size):
            kernel_labels = []
            kernel_values = values_matrix[:,j,i].reshape(values_matrix.shape[0],-1)
            for k in range(values_matrix.shape[0]):
                kernel = values_matrix[k,j,i].ravel()
                if mode == "scores":
                    kernel_label = ";".join(["{:.2e}".format(score) for score in kernel])
                elif mode == "weight":
                    kernel_label = ";".join(["{:.2e}".format(score) for score in kernel])
                elif mode == "masks":
                    nonzero_list = np.nonzero(kernel)[0].tolist()
                    kernel_label = ",".join([str(item) for item in nonzero_list]) if nonzero_list else "x"
                kernel_labels.append(kernel_label)
            match = np.prod(kernel_values == kernel_values[0,:].reshape(1,-1), axis=0)
            changed_values = kernel_values[:,match != 1].T.tolist()
            changed_pos = np.flatnonzero(match != 1).tolist()
            for values, pos in zip(changed_values, changed_pos):
                changes.append([f"row{i}, col{j}, pos{pos}", ' / '.join([str(item) for item in values])])
            labels[-1].append(" / ".join(kernel_labels))

    return [input_size, output_size], labels, changes

def monitor_mask_changes(dir_name, pkl_file):
    mode = "masks"
    if not pkl_file.endswith("pkl"):
        return
    print(pkl_file)
    output_dir = os.path.join(dir_name, pkl_file.replace(".pkl",""))
    os.makedirs(output_dir,exist_ok=True)
    changes_digest = []
    with open(os.path.join(dir_name, pkl_file), "rb") as f_in:
        logs = pickle.load(f_in)
        epoch_keys = list(logs.keys())
        epoch_keys.sort()
        layer_keys = list(logs[epoch_keys[0]].keys())
        for layer in layer_keys:
            print(layer)
            layer_logs = [logs[epoch][layer][mode] for epoch in epoch_keys]
            layer_sizes, labels, changes = preprocess_mask_data(layer_logs, mode)
            changes_digest.append([layer, len(changes), layer_logs[0].size, len(changes) / layer_logs[0].size])

            df = pd.DataFrame(labels,columns=[f"col{item}" for item in list(range(layer_sizes[1]))], index=[f"row{item}" for item in list(range(layer_sizes[0]))])
            with open(os.path.join(output_dir, f"{layer}.{mode}.html"), "w") as f_out:
                f_out.write(df.to_html())

            df2 = pd.DataFrame(changes,columns=["location", mode])
            with open(os.path.join(output_dir, f"{layer}.{mode}.changes.html"), "w") as f_out:
                f_out.write(df2.to_html())
                
        total_changes = np.sum([num_changes for [_, num_changes, _, _] in changes_digest])
        total_params = np.sum([num_params for [_, _, num_params, _] in changes_digest])
        print(f"found {total_changes}/{total_params} changes overall for the file {pkl_file}, ratio={round(total_changes/total_params, 3)}.")
        changes_digest.append(["total", total_changes, total_params, total_changes/total_params])
        df_all = pd.DataFrame(changes_digest,columns=["layer","number of changes", "total params", "ratio"])
        with open(os.path.join(output_dir, f"changes_overall.html"), "w") as f_out:
            f_out.write(df_all.to_html())


def monitor_bn_changes(dir_name, pkl_file):
    if not pkl_file.endswith("pkl"):
        return
    print(pkl_file)
    output_dir = os.path.join(dir_name, pkl_file.replace(".pkl",""))
    os.makedirs(output_dir,exist_ok=True)
    with open(os.path.join(dir_name, pkl_file), "rb") as f_in:
        logs = pickle.load(f_in)
        epoch_keys = list(logs.keys())
        epoch_keys.sort()
        layer_keys = list(logs[epoch_keys[0]].keys())
        for layer in layer_keys:
            print(layer)
            layer_means = [logs[epoch][layer]["mean"] for epoch in epoch_keys]
            layer_vars = [logs[epoch][layer]["var"] for epoch in epoch_keys]

            labels = layer_means + layer_vars
            columns = [f"col{item}" for item in range(len(labels[0]))]
            rows = [f"mean-{epoch}" for epoch in epoch_keys] + [f"var-{epoch}" for epoch in epoch_keys]

            df = pd.DataFrame(labels,columns=columns, index=rows)
            with open(os.path.join(output_dir, f"{layer}.html"), "w") as f_out:
                f_out.write(df.to_html())
                