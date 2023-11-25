import os

import numpy as np

from .get_functions import get_save_path

def save_metrics(args, test_results, model_dirs, current_epoch, current_trial):
    print("###################### TEST REPORT ######################")
    for metric in test_results.keys():
        print("Mean {}    :\t {}".format(metric, test_results[metric]))
    print("###################### TEST REPORT ######################\n")

    if args.train_data_type == args.test_data_type:
        test_results_save_path = os.path.join(model_dirs, 'test_reports', 'test_report(EPOCH {})_Trial{}.txt'.format(current_epoch, current_trial))
    else:
        test_results_save_path = os.path.join(model_dirs, 'test_reports', 'Generalizability test_reports({}->{})_Trial{}.txt'.format(args.train_data_type, args.test_data_type, current_trial))

    f = open(test_results_save_path, 'w')

    f.write("###################### TEST REPORT ######################\n")
    for metric in test_results.keys():
        f.write("Mean {}    :\t {}\n".format(metric, test_results[metric]))
    f.write("###################### TEST REPORT ######################\n")

    f.close()

    print("test results txt file is saved at {}".format(test_results_save_path))

def save_total_trial_binary_segmentation(args):
    model_dirs = get_save_path(args)

    total_metrics_dict = dict()

    for metric in args.metric_list:
        total_metrics_dict[metric] = list()

    for current_trial in range(1, args.num_trial + 1):
        print("Loading {} Trial results...".format(current_trial))

        if args.train_data_type == args.test_data_type:
            load_results_file = os.path.join(model_dirs, 'test_reports', 'test_report(EPOCH {})_Trial{}.txt'.format(args.final_epoch, current_trial))
        else:
            load_results_file = os.path.join(model_dirs, 'test_reports', 'Generalizability test_reports({}->{})_Trial{}.txt'.format(args.train_data_type, args.test_data_type, current_trial))

        f = open(load_results_file)
        while True:
            line = f.readline()
            if not line: break

            if line.split()[1] in args.metric_list: total_metrics_dict[line.split()[1]].append(float(line.split()[-1]))

        f.close()

    print("###################### TEST REPORT ######################")
    for metric in total_metrics_dict.keys():
        print("Trial Mean {}    :\t {} ({}) [{} | {} | {}]".format(metric,
                                                                   np.round(np.mean(total_metrics_dict[metric]), 4),
                                                                   np.round(np.std(total_metrics_dict[metric]), 4),
                                                                   np.round(total_metrics_dict[metric][0], 4),
                                                                   np.round(total_metrics_dict[metric][1], 4),
                                                                   np.round(total_metrics_dict[metric][2], 4)))
    print("###################### TEST REPORT ######################\n")

    if args.train_data_type == args.test_data_type:
        test_results_save_path = os.path.join(model_dirs, 'test_reports', 'test_report(EPOCH {})_TotalResults.txt'.format(args.final_epoch))
    else:
        test_results_save_path = os.path.join(model_dirs, 'test_reports',  'Generalizability test_reports({}->{})_TotalResults.txt'.format(args.train_data_type, args.test_data_type))

    f = open(test_results_save_path, 'w')

    f.write("###################### TEST REPORT ######################\n")
    for metric in total_metrics_dict.keys():
        f.write("Trial Mean {}    :\t {} ({}) [{} | {} | {}]\n".format(metric,
                                                                   np.round(np.mean(total_metrics_dict[metric]), 4),
                                                                   np.round(np.std(total_metrics_dict[metric]), 4),
                                                                   np.round(total_metrics_dict[metric][0], 4),
                                                                   np.round(total_metrics_dict[metric][1], 4),
                                                                   np.round(total_metrics_dict[metric][2], 4)))
    f.write("###################### TEST REPORT ######################\n")

    f.close()

    print("test results txt file is saved at {}".format(test_results_save_path))