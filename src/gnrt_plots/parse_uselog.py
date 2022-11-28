import json
import os
import collections
import csv
import matplotlib.pyplot as plt
import pandas as pd
import glob
import statistics

def count_rules(info):
    rules = 0
    for feat in info:
        for fv in info[feat]:
            rules += len(info[feat][fv])
    return rules

def cmptmax(files, key):
    maxkey = 0

    fit = 0

    nof_dt = 0
    for file in files:
        with open(file, 'r') as f:
            keydict = json.load(f)

        for dt in keydict['stats']:
            k = keydict['stats'][dt][key]
            nof_dt += 1

            if k <= 10:
                fit += 1

            if k > maxkey:
                maxkey = k

    print(f'nof_dt: {nof_dt}')
    print(f'fit: {fit}')
    return maxkey

def parse_log(log, model, dictnry):
    with open(log, 'r') as f:
        lines = f.readlines()

    if model == 'dl':
        inst_mark = 'inst:'
        expl_mark = 'expl:'
        size_mark = 'hypos left:'
        use_mark = 'used rule:'
        use_size_mark = 'used rule hypos left:'
    elif model == 'bt':
        inst_mark = 'explaining:'
        expl_mark = 'explanation:'
        size_mark = 'hypos left:'
        use_mark = 'used rule:'
        use_size_mark = 'used rule hypos left:'
    elif model == 'bnn':
        inst_mark = 'explaining:'
        expl_mark = 'explanation:'
        size_mark = 'explanation size:'
        use_mark = 'used rule:'
        use_size_mark = 'used rule size:'
    else:
        # print('model wrong')
        exit(1)

    """
    
    compute the average number of rules 
    used to reduce an explanation size
    
    """

    expl_ids = []

    for i, line in enumerate(lines):
        if expl_mark in line:
            if use_mark in lines[i + 2]:
                expl_ids.append(i)

    expl_ids.append(len(lines))

    bgs = []
    nof_rules = []
    for i, start in enumerate(expl_ids[:-1]):
        end = expl_ids[i + 1]
        bgs.append([])
        for id in range(start, end):
            line = lines[id]
            if use_size_mark in line:
                bgs[-1].append(int(line.split(use_size_mark)[-1]))
        nof_rules.append(len(bgs[-1]))

    assert len(expl_ids) - 1 == len(bgs)

    dtname = log.rsplit(f'{model}_q6_')[-1].split('_test1')[0]


    """
    
    Distribution of used rules
    
    """

    header = ['lits', 'count', 'dist']
    stats = []

    sizes = [size for bg in bgs for size in bg]
    lits_count = collections.Counter(sizes)

    total_count = sum(lits_count.values())

    for lit in sorted(lits_count.keys()):
        count = lits_count[lit]
        dist = round(count / total_count, 3)
        stats.append([lit, count, dist])

    saved_dir = f'../stats/usedist/{model}'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)
    fn = f'{saved_dir}/{dtname}.csv'

    with open(fn, 'w') as f:
        write = csv.writer(f)
        write.writerow(header)
        write.writerows(stats)

    return fn

def scatter(file):
    model = 'dl' if '_dl' in file else 'bt'
    df = pd.read_csv(file)
    lits = df['lits'].to_list()
    counts = df['count'].to_list()
    plt.figure(figsize=(8, 16))
    plt.scatter(lits, counts, c='green', s=70, alpha=0.1)
    plt.title("Lending.csv Useful Rules in {0}".format(model), fontsize=18, y=1.03)
    plt.xlabel("Lits", fontsize=14, labelpad=15)
    plt.ylabel("Count", fontsize=14, labelpad=15)
    plt.tick_params(labelsize=12, pad=6)

    plt.savefig(file[ : file.rfind('.')] + '.pdf')

def user_table(m2csvs, m2logs):

    datasets = sorted(['adult', 'compas', 'lending', 'recidivism'])
    feats = {'adult': 65, 'compas': 16, 'lending': 35, 'recidivism': 29}

    dt2dist = { dt: collections.defaultdict(lambda : {i: '-' for i in range(1, 8)})
                for dt in datasets}

    for m in m2csvs:
        for file in m2csvs[m]:
            dt = file.rsplit('/')[-1].rsplit('.csv')[0]
            with open(file, 'r') as f:
                lines = list(map(lambda l: l.split(','), f.readlines()[1:]))

            for line in lines:
                lits = int(line[0])
                if lits in range(1, 8):
                    dist = float(line[2]) * 100
                    dt2dist[dt][m][lits] = round(dist, 1)

    head = [
        '\\begin{table}[ht!]',
        '\\centering',
        '\\caption{Size Distribution Of Used Rules in DLs, BTs and BNNs}',
        '\\label{tab:usedr}',
        '\\scalebox{0.85}{',
        '\\begin{tabular}{cccccccccc}\\toprule',
        '	\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Feats} & \\multirow{2}{*}{Model} & \\multicolumn{7}{c}{Distribution (\\%)} \\\\ \\cmidrule{4-10}',
        '				 & & & $1$ & $2$ & $3$ & $4$ & $5$ & $6$ & $7$ \\\\ \\midrule ']

    rows = []
    end = [
        '\end{tabular}',
        '}',
        '\\end{table}']

    models = ['dl', 'bt', 'bnn']

    for dt in datasets:
        for i, m in enumerate(models):
            c1_c2 = ['', ''] if i in [0, 2] else [dt, f'${feats[dt]}$']
            others = [m.upper()] + [f'${dt2dist[dt][m][lits]}$' for lits in range(1, 8)]
            row = ' & '.join(c1_c2 + others) + ' \\\\'
            if i == 2:
                if dt != datasets[-1]:
                    row += '\\midrule'
                else:
                    row += '\\bottomrule'
            rows.append(row)

    saved_dir = '../stats/usedist/table'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    fn = f'{saved_dir}/userules_table.txt'

    with open(fn, 'w') as f:
        f.writelines('\n'.join(head + rows + end))

if __name__ == '__main__':

    m2logs = collections.defaultdict(lambda: [])

    for root, dirs, files in os.walk('../logs'):
        for file in files:
            if '/rextract' in root:
                continue
            if 'userules' in file:
                m = root.split('logs/')[-1].split('/')[0]
                m2logs[m].append(os.path.join(root, file))

    dictnry = {model: {'stats': {},
                       'preamble': {'benchmark': model.upper(),
                                    'prog_arg': model.upper(),
                                    'program': model.upper(),
                                    'prog_alias': model.upper()}} for model in m2logs}

    m2csvs = {m: [] for m in m2logs}
    for m in m2logs:
        for log in m2logs[m]:
            csvfn = parse_log(log, m, dictnry)
            m2csvs[m].append(csvfn)
    """
    
    useful rule size distribution (table)
    
    """

    user_table(m2csvs, m2logs)
