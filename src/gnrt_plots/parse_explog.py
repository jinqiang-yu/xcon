import os
import sys
import collections
import glob
import json
import statistics

def cmptmax(files, key):
    maxkey = 0
    for file in files:
        with open(file, 'r') as f:
            keydict = json.load(f)

        for dt in keydict['stats']:
            k = keydict['stats'][dt][key]
            if k > maxkey:
                maxkey = k
    return maxkey

def extract_exps(logs, model, appr):
    stats_dict = collections.defaultdict(
        lambda: {"preamble": {"program": None, "prog_args": None, "prog_alias": None, "benchmark": None},
                 "stats": collections.defaultdict(lambda: {'status': True })})

    for log in logs:
        dtname = log.rsplit('/')[-1].split('_', maxsplit=1)[-1].split('_test1')[0]
        xtype = 'abd' if '_abd' in log else 'con'
        bg = 'size5' if '_size5' in log else 'ori'
        conf = tuple([bg, xtype])

        if stats_dict[conf]['preamble']['program'] is None:
            for e in ['program', 'prog_args', 'prog_alias']:
                stats_dict[conf]['preamble'][e] = "{0}$_{1}$".format(appr, '{' + model + '}') if bg == 'ori' \
                                                    else "{0}$^r_{1}$".format(appr, '{' + model + '}')

        with open(log, 'r') as f:
            lines = f.readlines()
        lines = list(filter(lambda l: len(l.strip()) > 0, lines))
        lines = list(map(lambda l: l.strip(), lines))

        for i in range(len(lines)-1, -1, -1):
            if 'exptimes:' in lines[i]:
                exptimes = lines[i].split('exptimes:')[-1].strip().strip('[').strip(']').split(',')
                exptimes = list(map(lambda l: float(l), exptimes))
        insts = []

        if model == 'dl':
            inst_mark = 'inst:'
            expl_mark = 'expl:'
            size_mark = 'hypos left:'
        elif model == 'bt':
            inst_mark = 'explaining:'
            expl_mark = 'explanation:'
            size_mark = 'hypos left:'
        elif model == 'bnn':
            inst_mark = 'explaining:'
            expl_mark = 'explanation:'
            size_mark = 'explanation size:'
        else:
            #print('model wrong')
            exit(1)

        for i, line in enumerate(lines):
            if inst_mark in line:
                insts.append(i)

        insts.append(len(lines))

        for i, start in enumerate(insts[:-1]):
            end = insts[i + 1]
            inst = lines[insts[i]].split(inst_mark)[-1].strip()[:-1].strip('"')
            dt_inst = f'{dtname}_{inst}'
            expls = []
            sizes = []
            for ii in range(start + 1, end):
                if expl_mark in lines[ii]:
                    expl = lines[ii].split(expl_mark)[-1].strip().strip('"')
                    expls.append(expl)
                    for iii in range(ii + 1, end):
                        if size_mark in lines[iii]:
                            sizes.append(int(lines[iii].split(size_mark)[-1]))
                            break

            if len(expls) == 0:
                assert len(sizes) == 0
                #expls.append(inst)
                #sizes.append(len(inst.split('AND')))
                expls.append(None)
                sizes.append(1)

            assert len(expls) == len(sizes)
            stats_dict[conf]['stats'][dt_inst]['expl'] = expls[0]
            stats_dict[conf]['stats'][dt_inst]['minsize'] = sizes[0]
            stats_dict[conf]['stats'][dt_inst]['totrtime'] = exptimes[i] * 1000
            stats_dict[conf]['stats'][dt_inst]['exprtime'] = exptimes[i] / len(expls) * 1000

    for conf in stats_dict:
        bg, xtype = conf
        saved_dir = f'../stats/{f"{appr}_" if appr != "xcon" else ""}expls/{model}'
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)

        jsonfn = f'{saved_dir}/{bg}_{xtype}.json'
        with open(jsonfn, 'w') as f:
            json.dump(stats_dict[conf], f, indent=4)

def grt_scatter(files, model, xtype, appr):
    keys = ['exprtime']
    for key in keys:
        maxkey = cmptmax(files, key)
        maxkey = maxkey if maxkey >= 1 else 1

        saved_dir = '../plots/{0}{1}'.format(f'{appr}/' if appr != 'xcon' else '', model)
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)
        pdffn = f'{saved_dir}/{model}_{key}_{xtype}_scatter.pdf'

        cmd = 'python ./gnrt_plots/mkplot/mkplot.py ' \
              '-l -p scatter -b pdf --save-to {0} -k {1} --shape squared --font-sz 26 -a 0.1 --tol-loc none ' \
              ' -t {2} --ylog --ymax {2} --ymin 0.1 --xlog {3}'.format(pdffn, key,
                                                                       maxkey, ' '.join(files))
        #print('\ncmd:\n{0}'.format(cmd))
        os.system(cmd)
        #print()

def grt_size_scatter(files, model, xtype, appr):
    keys = ['minsize']
    for key in keys:
        maxkey = cmptmax(files, key)
        maxkey = maxkey if maxkey >= 1 else 1

        if appr == 'xcon':
            if '/bnn/' in files[0]:
                maxkey = 17
            elif '/dl/' in files[0]:
                maxkey = 15
            else:
                maxkey = 15
        else:
            if '/bnn/' in files[0]:
                maxkey = 16
            elif '/dl/' in files[0]:
                maxkey = 15
            else:
                maxkey = 12

        saved_dir = '../plots/{0}{1}'.format(f'{appr}/' if appr != 'xcon' else '', model)
        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)

        pdffn = f'{saved_dir}/{model}_{key}_{xtype}_scatter.pdf'

        cmd = 'python ./gnrt_plots/mkplot/mkplot.py ' \
              '-l -p scatter -b pdf --save-to {0} -k {1} --shape squared --font-sz 25 --tol-loc none ' \
              ' -t {2} --ymax {2} {3}'.format(pdffn, key,
                                                                       maxkey, ' '.join(files))
        #print('\ncmd:\n{0}'.format(cmd))
        os.system(cmd)
        #print()

def avg_minsize(m2jsons, datasets):
    change = {'before': {}, 'after': {}}
    change = {dt: collections.defaultdict(lambda : []) for dt in datasets}

    for m in m2jsons:
        for file in m2jsons[m]:
            xtype = 'abd' if '_abd' in file else 'con'
            kngl = 'without' if 'ori_'in file else 'with'
            with open(file, 'r') as f:
                info = json.load(f)

            dt_inst_list = set(info['stats'].keys())

            for dt in datasets:
                conf = tuple([m, xtype, kngl])
                q_dt = 'q6_' + dt

                for dt_inst in dt_inst_list:
                    if q_dt in dt_inst:
                        minsize = info['stats'][dt_inst]['minsize']
                        change[dt][conf].append(minsize)
    for dt in change:
        for conf in change[dt]:
            change[dt][conf] = round(statistics.mean(change[dt][conf]), 2)
            if change[dt][conf] % 1 == 0:
                change[dt][conf] = f'{int(change[dt][conf])}.00'
            else:
                change[dt][conf] = '{0:.2f}'.format(change[dt][conf])

    return change

def expls_table(change, datasets):
    feats = {'adult': 65, 'compas': 16, 'lending': 35, 'recidivism': 29}

    head = ['\\begin{table}[ht!]',
            '\\centering',
            '\\caption{Change of Average Minimum Explanation Size}',
            '\\label{tab:exp}',
            '\\scalebox{0.86}{',
            '\\begin{tabular}{ccccccc}\\toprule',
            '	\\multirow{2}{*}{Dataset} & \\multirow{2}{*}{Feats}  & \\multirow{2}{*}{Model} & '
            '\\multicolumn{2}{c}{AXp Size} & ',
            '	\\multicolumn{2}{c}{CXp Size}  \\\\ \\cmidrule{4-7}',
            '				 & & & Before & After & Before & After  \\\\ \\midrule ']

    rows = []

    end = [
            '\\end{tabular}',
            '}',

            '\\end{table}']

    models = ['dl', 'bt', 'bnn']
    for dt in datasets:
        for i, m in enumerate(models):
            c1_c2 = ['', ''] if i in [0, 2] else [dt, f'${feats[dt]}$']
            others = [m.upper()]

            for xtype in ['abd', 'con']:
                for kngl in ['without', 'with']:
                    others.append(f'${change[dt][m, xtype, kngl]}$')
            row = ' & '.join(c1_c2 + others) + ' \\\\'
            if i == 2:
                if dt != datasets[-1]:
                    row += '\\midrule'
                else:
                    row += '\\bottomrule'
            rows.append(row)

    saved_dir = '../stats/expls/table'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    with open(f'{saved_dir}/expls_table.txt', 'w') as f:
        f.writelines('\n'.join(head + rows + end))


if __name__ == '__main__':

    m2logs = collections.defaultdict(lambda : [])
    appr = sys.argv[-1]
    for root, dirs, files in os.walk('../{0}logs'.format(f'{appr}_' if appr != 'xcon' else '')):
        for file in files:
            if '/rextract' in root or '/hexp' in root:
                continue
            if file.endswith('log') and 'userules' not in file:
                m = root.split('logs/')[-1].split('/')[0]
                m2logs[m].append(os.path.join(root, file))

    """
    
    parse logs to json files
    
    """
    for m in m2logs:
        extract_exps(m2logs[m], m, appr)

    """
    
    generating plots
    
    """

    files = {}

    for m in m2logs:
        json_files = glob.glob('../stats/{0}expls/{1}/*.json'.format(f'{appr}_' if appr != 'xcon' else '',
                                                               m))
        bg = ['ori', 'size5']
        xtypes = ['abd', 'con']
        files[m] = {}
        for xtype in xtypes:
            files[m][xtype] = sorted(filter(lambda l: xtype in l, json_files), reverse=True)

    for m in files:
        for xtype in files[m]:
            grt_scatter(files[m][xtype], m, xtype, appr)
            grt_size_scatter(files[m][xtype], m, xtype, appr)

    """
    Generate table    
    """

    if appr == 'xcon':
        m2jsons = collections.defaultdict(lambda: [])
        for root, dirs, files in os.walk('../stats/expls/'):
            for file in files:
                if file.endswith('.json'):
                    m = root.split('expls/')[-1].strip()
                    m2jsons[m].append(os.path.join(root, file))

        datasets = sorted(['adult', 'compas', 'lending', 'recidivism'])
        change = avg_minsize(m2jsons, datasets)
        expls_table(change, datasets)
    exit()