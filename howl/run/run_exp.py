from .args import ArgumentParserBuilder, opt
import random
import os
import subprocess
import numpy as np

from datetime import datetime
from openpyxl import Workbook, load_workbook


def get_col(char, ind):
    return chr(ord(char) + ind)


def main():
    apb = ArgumentParserBuilder()
    apb.add_options(opt('--n',
                        type=int,
                        default=1,
                        help='number of experiments to run'),
                    opt('--dataset_path',
                        type=str,
                        default="/data/speaker-id-split-medium"),
                    opt('--noiseset_path',
                        type=str,
                        default="/data/MS-SNSD"))

    args = apb.parser.parse_args()

    # Create new workbook
    wb = Workbook()
    now = datetime.now()
    clean_sheet = wb.create_sheet('clean', 0)
    noisy_sheet = wb.create_sheet('noisy', 1)

    model_types = ['res8', 'las', 'lstm', 'mobilenet']
    col_map = {
        'Dev positive': 'B',
        'Dev noisy positive': 'B',
        'Dev negative': 'H',
        'Dev noisy negative': 'H',
        'Test positive': 'N',
        'Test noisy positive': 'N',
        'Test negative': 'T',
        'Test noisy negative': 'T'
    }
    metrics = ["threshold", "tp", "tn", "fp", "fn"]

    clean_col_names = ["Dev positive", "Dev negative", "Test positive", "Test negative"]
    noisy_col_names = ["Dev noisy positive", "Dev noisy negative", "Test noisy positive", "Test noisy negative"]

    clean_cols_aggregated = {}
    noisy_cols_aggregated = {}
    
    for col in clean_col_names:
        ind = col_map[col] + '1'
        clean_sheet[ind] = col

        for i, metric in enumerate(metrics):
            ind = get_col(col_map[col], i) + '2'
            clean_sheet[ind] = metric

            clean_cols_aggregated[get_col(col_map[col], i)] = []

    for col in noisy_col_names:
        ind = col_map[col] + '1'
        noisy_sheet[ind] = col

        for i, metric in enumerate(metrics):
            ind = get_col(col_map[col], i) + '2'
            noisy_sheet[ind] = metric

            noisy_cols_aggregated[get_col(col_map[col], i)] = []



    clean_sheet['A3'] = "mean"
    clean_sheet['A4'] = "std"
    clean_sheet['A5'] = "p90"
    clean_sheet['A6'] = "p95"
    clean_sheet['A7'] = "p99"
    clean_sheet['A8'] = "sum"

    noisy_sheet['A3'] = "mean"
    noisy_sheet['A4'] = "std"
    noisy_sheet['A5'] = "p90"
    noisy_sheet['A6'] = "p95"
    noisy_sheet['A7'] = "p99"
    noisy_sheet['A8'] = "sum"


    os.system("mkdir -p exp_results")
    dt_string = now.strftime("%b-%d-%H-%M")
    file_name = "exp_results/hey_ff_" + dt_string + ".xlsx"
    print("exp_result stored at ", file_name)


    os.environ["DATASET_PATH"] = args.dataset_path
    os.environ["WEIGHT_DECAY"] = "0.00001"
    os.environ["NUM_EPOCHS"] = "300"
    os.environ["LEARNING_RATE"] = "0.001"
    os.environ["LR_DECAY"] = "0.98"
    os.environ["BATCH_SIZE"] = "16"
    os.environ["MAX_WINDOW_SIZE_SECONDS"] = "0.5"
    os.environ["USE_NOISE_DATASET"] = "True"
    os.environ["NUM_MELS"] = "40"
    os.environ["INFERENCE_SEQUENCE"] = "[0,1,2]"

    os.environ["VOCAB"] = '[" hey","fire","fox"]'
    os.environ["NOISE_DATASET_PATH"] = args.noiseset_path
    os.environ["INFERENCE_THRESHOLD"] = "0"


    for i in range(args.n):
        print("\titeration: ", i , " - ", datetime.now().strftime("%H-%M"), flush=True)
        os.environ["SEED"] = str(random.randint(1,1000000))

        row_index = str(i + 10)
        clean_sheet['A'+row_index] = str(i)
        noisy_sheet['A'+row_index] = str(i)

        workspace_path = os.getcwd() + "/workspaces/exp_hey_ff_res8/" + str(i)
        os.system("mkdir -p " + workspace_path)

        exp_execution = os.system("python -m howl.run.train --model res8 --workspace " + workspace_path + "  -i " + args.dataset_path)

        log_path = workspace_path + "/results.csv"
        raw_log = subprocess.check_output(['tail', '-n', '8', log_path]).decode("utf-8") 
        logs = raw_log.split('\n')

        for log in logs:
            if len(log) == 0:
                break
            vals = log.split(',')
            key = vals[0]
            start_col = col_map[key]

            for i, metric in enumerate(vals[1:]):
                ind = get_col(start_col, i) + str(row_index)

                if "noisy" in key:
                    noisy_sheet[ind] = metric
                    noisy_cols_aggregated[get_col(start_col, i)].append(float(metric))
                else:
                    clean_sheet[ind] = metric
                    clean_cols_aggregated[get_col(start_col, i)].append(float(metric))

        for idx, agg_results in clean_cols_aggregated.items():
            results = np.array(agg_results)
            clean_sheet[idx + '3'] = str(results.mean())
            clean_sheet[idx + '4'] = str(results.std())
            clean_sheet[idx + '5'] = str(np.percentile(results, 90))
            clean_sheet[idx + '6'] = str(np.percentile(results, 95))
            clean_sheet[idx + '7'] = str(np.percentile(results, 99))
            clean_sheet[idx + '8'] = str(results.sum())

        for idx, agg_results in noisy_cols_aggregated.items():
            results = np.array(agg_results)
            noisy_sheet[idx + '3'] = str(results.mean())
            noisy_sheet[idx + '4'] = str(results.std())
            noisy_sheet[idx + '5'] = str(np.percentile(results, 90))
            noisy_sheet[idx + '6'] = str(np.percentile(results, 95))
            noisy_sheet[idx + '7'] = str(np.percentile(results, 99))
            noisy_sheet[idx + '8'] = str(results.sum())


        wb.save(file_name)



if __name__ == '__main__':
    main()
