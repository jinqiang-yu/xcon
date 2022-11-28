import os
import sys
from options import Options
import json
import collections
import glob

def allrules(cmpr_dict, q2dt):
    allr = []
    for q in sorted(q2dt.keys(), reverse=True):
        h = ['\\paragraph{' + q[1] + ' Intervals}']
        hhh = ['\\begin{center}',

               '\\begin{tabular}{|p{0.25\\linewidth} |p{0.04\\linewidth} |' \
               ' p{0.04\\linewidth} | p{0.04\\linewidth} |  p{0.08\\linewidth} |' \
               ' p{0.04\\linewidth} | p{0.04\\linewidth} | p{0.04\\linewidth} |' \
               ' p{0.08\\linewidth} | p{0.1\\linewidth} |} \\hline',

               ' \\multirow{2}{*}{\\textbf{Dataset}} & \\multicolumn{4}{c}{\\textbf{Min Size}} &'
               ' \\multicolumn{4}{|c|}{\\textbf{Nof Expls}} & \\textbf{Features} \\\\ \\cline{2-9}',
               ' & $\\downarrow$ & $==$ & $\\uparrow$ & Change & $\\downarrow$ & $==$ &'
               ' $\\uparrow$ & Change & \\textbf{Union} \\\\ \\hline']

        e = ['\\end{tabular}',
             ' \\end{center}']
        xtype2b = collections.defaultdict(lambda : [])

        for dt in q2dt[q]:
            name = dt if dt.endswith('.csv') else dt[:dt.rfind('.')]
            name = name.replace("_", "\\_")
            for xtype in cmpr_dict[dt]:
                conf = list(filter(lambda l: 'size' in l, cmpr_dict[dt][xtype].keys()))[0]
                stopp = list(filter(lambda l: 'size' not in l, cmpr_dict[dt][xtype][conf].keys()))[0]
                stat = cmpr_dict[dt][xtype][conf][stopp]

                min_size = stat['min_size']
                avg_min_size = list(map(lambda l: float(l), stat['avg_min_size'].split('to')))

                nof_expls = stat['nof_expls']
                avg_nof_expls = list(map(lambda l: float(l), stat['avg_nof_expls'].split('to')))

                avg_feats = list(map(lambda l: float(l), stat['avg_feats'].split('to')))

                b = f' {name} &' \
                    f' {min_size["decrease"]} & {min_size["equal"]} & {min_size["increase"]} &' \
                    f' {avg_min_size[0]} $\\rightarrow$ {avg_min_size[1]} &' \
                    f' {nof_expls["decrease"]} & {nof_expls["equal"]} & {nof_expls["increase"]} &' \
                    f' {avg_nof_expls[0]} $\\rightarrow$ {avg_nof_expls[1]} &' \
                    f' {avg_feats[0]} $\\rightarrow$ {avg_feats[1]} \\\\ \\hline'

                xtype2b[xtype].append(b)

        whole = h[:]
        for xtype in ['abd', 'con']:
            hh = '\\paragraph{xtype}'.replace('xtype', 'AXp' if xtype == 'abd' else 'CXp')
            whole.append(hh)
            whole.extend(hhh)
            b = xtype2b[xtype]
            whole.extend(b)
            whole.extend(e)
            whole.append('')

        allr.append('\n'.join(whole))

    with open('compr/allr.txt', 'w') as f:
        f.write('\n'.join(allr))

def sizestop(cmpr_dict, q2dt):
    sizes = []


    for q in sorted(q2dt.keys(), reverse=True):
        if q == 'q4':
            continue
        h = ['\\paragraph{' + q[1] + ' Intervals}']
        hhh = ['\\begin{center}',

               '\\begin{tabular}{|p{0.1\\linewidth} |p{0.1\\linewidth} |p{0.04\\linewidth} |' \
               ' p{0.04\\linewidth} | p{0.04\\linewidth} |  p{0.08\\linewidth} |' \
               ' p{0.04\\linewidth} | p{0.04\\linewidth} | p{0.04\\linewidth} |' \
               ' p{0.1\\linewidth} | p{0.1\\linewidth} |} \\hline',

               ' \\multirow{2}{*}{\\textbf{Expl}} & \\multirow{2}{*}{\\textbf{Size}} & \\multicolumn{4}{c}{\\textbf{Min Size}} &'
               ' \\multicolumn{4}{|c|}{\\textbf{Nof Expls}} & \\textbf{Features} \\\\ \\cline{3-10}',

               ' & & $\\downarrow$ & $==$ & $\\uparrow$ & Change & $\\downarrow$ & $==$ &'
               ' $\\uparrow$ & Change & \\textbf{Union} \\\\ \\hline']

        e = ['\\end{tabular}',
             ' \\end{center}']

        whole = h[:]

        for dt in q2dt[q]:
            name = dt if dt.endswith('.csv') else dt[:dt.rfind('.')]
            name = name.replace("_", "\\_")
            hh = '\paragraph{' + name + '}'
            bs = []
            for xtype in cmpr_dict[dt]:
                conf = list(filter(lambda l: 'size' in l, cmpr_dict[dt][xtype].keys()))[0]

                stopps = list(filter(lambda l: 'armine' not in l, cmpr_dict[dt][xtype][conf].keys()))
                #stopps = list(filter(lambda l: l != 'size300', stopps))

                stopps.sort(key= lambda l: int(l.replace('size', '')[:-2]), reverse=True)
                stopps.insert(0, 'armine')
                for i, stopp in enumerate(stopps):

                    stat = cmpr_dict[dt][xtype][conf][stopp]

                    min_size = stat['min_size']
                    avg_min_size = list(map(lambda l: float(l), stat['avg_min_size'].split('to')))

                    nof_expls = stat['nof_expls']
                    avg_nof_expls = list(map(lambda l: float(l), stat['avg_nof_expls'].split('to')))

                    avg_feats = list(map(lambda l: float(l), stat['avg_feats'].split('to')))

                    expl = 'AXp' if xtype == 'abd' else 'CXp'

                    c0 = '\\multirow{' + str(len(stopps)) + '}{*}{' + expl + '}' if stopp == 'armine' else ''
                    c1 = '$\\leq {0}$'.format(stopp.replace('size', '')[:-2]) if stopp != 'armine' else 'all'

                    c9 = "\\hline" if i == len(stopps) - 1 else "\\cline{2-11}"

                    b = f' {c0} & {c1} &' \
                        f' ${min_size["decrease"]}$ & ${min_size["equal"]}$ & ${min_size["increase"]}$ &' \
                        f' ${avg_min_size[0]}$ $\\rightarrow$ ${avg_min_size[1]}$ &' \
                        f' ${nof_expls["decrease"]}$ & ${nof_expls["equal"]}$ & ${nof_expls["increase"]}$ &' \
                        f' ${avg_nof_expls[0]}$ $\\rightarrow$ ${avg_nof_expls[1]}$ &' \
                        f' ${avg_feats[0]}$ $\\rightarrow$ ${avg_feats[1]}$ \\\\ ' \
                        f'{c9}'

                    bs.append(b)

            whole.append(hh)
            whole.extend(hhh)
            whole.extend(bs)
            whole.extend(e)

        sizes.append('\n'.join(whole))

    with open('compr/sizestop.txt', 'w') as f:
        f.write('\n'.join(sizes))

def tojson(jsonfn):

    with open(jsonfn, 'r') as f:
        stat_dict = json.load(f)

    for dataset in stat_dict:
        q = dataset.split('_')[0].strip()
        for xtype in stat_dict[dataset]:
            for stopp in stat_dict[dataset][xtype]:
                alias = stopp.replace('size', '')[:-2] if 'size' in stopp else stopp
                stopp_dict = {'stats' : {},
                              'preamble' : {'benchmark': q,
                                            'prog_arg': None,
                                            'program': alias,
                                            'prog_alias': alias}}

                cstat = stat_dict[dataset][xtype][stopp]
                for inst in cstat:
                    stopp_dict['stats'][inst] = {'status': True}
                    stopp_dict['stats'][inst]['nof_expls'] = cstat[inst]['nof_expls']
                    stopp_dict['stats'][inst]['min_size'] = cstat[inst]['min_size']

                with open(jsonfn[: jsonfn.rfind('/') + 1] + f'visualisation/{dataset}_{xtype}_{stopp}.json', 'w') as f:
                    json.dump(stopp_dict, f, indent=4)
                    f.close()

def scatter(jsonpos):
    jsons = glob.glob(jsonpos + '/*json')

    dt_xtypes = sorted(set(map(lambda l: l.split('_abd_')[0] + '_abd_' if '_abd_' in l
                                else l.split('_con_')[0] + '_con_', jsons)), reverse=True)

    for a in ['size300', 'size400', 'size500']:
        stopps = ['org'] + [a]

        for s in ['nof_expls', 'min_size']:
            for dt_xtype in dt_xtypes:
                if '/q4_' in dt_xtype:
                    continue
                max_s = set()
                for stopp in stopps:
                    fn = dt_xtype + stopp + '.json'
                    with open(fn, 'r') as f:
                        stat = json.load(f)

                    s_set = set()
                    for inst in stat['stats']:
                        s_set.add(stat['stats'][inst][s])
                    max_s.add(max(s_set))

                max_s = max(max_s)
                jsonfns = [dt_xtype + stopp + '.json' for stopp in stopps]

                pdffn = jsonpos + '/plot/' + dt_xtype[dt_xtype.rfind('/') + 1:] + f'{stopps[1]}_{s}.pdf'

                command = 'python3 ../../../rc2/mkplot-master/mkplot.py -l -p scatter ' \
                          f'-b pdf --save-to {pdffn} -k {s} --shape squared ' \
                          f'-t {max_s} --ymax {max_s + 0.1 * max_s} --ymin 0.9 ' \
                          f'{" ".join(jsonfns)}'

                print('command:', command)
                os.system(command)

def sct2latex(plotpos):
    plots = glob.glob(plotpos + '/*pdf')
    plots = list(filter(lambda l: 'size500' in l and 'q6_' in l, plots))
    plots = list(map(lambda l: 'sizeplot/' + l.split('/plot/')[-1], plots))

    dt2plots = collections.defaultdict(lambda : [None, None, None, None])
    for p in plots:
        xtype = 'abd' if '_abd_' in p else 'con'
        s = p.split('size500')[-1].strip('_').replace('.pdf', '')
        pos = 0 if xtype == 'abd' else 2
        pos = pos if s == 'min_size' else pos + 1
        dt = p.split(f'_{xtype}')[0].replace('sizeplot/', '')
        dt2plots[dt][pos] = p

    datasets = sorted(dt2plots.keys(), reverse=True)

    whole = []
    for dt in datasets:
        h = ['\\begin{figure*}[!b]',
            ' \\centering']
        bs = []
        e = ['\\caption{' + dt.replace('_', '\\_') + '}',
            '  \\end{figure*}']
        for i, p in enumerate(dt2plots[dt]):
            xtype = 'AXp' if '_abd_' in p else 'CXp'
            s = ' '.join(p.split('size500')[-1].strip('_').replace('.pdf', '').split('_'))

            b = ['\\begin{subfigure}[b]{0.24\\textwidth}',
                '    \\centering',
                '    \\includegraphics[width=\\textwidth]{' + p + '}',
                '    \\caption{' + ' ' + xtype + s + '}',
                '  \\end{subfigure}%']
            bs.extend(b)

        dtfig = h[:]
        dtfig.extend(bs)
        dtfig.extend(e)

        whole.extend(dtfig)


    #options.model[: options.model.rfind('/') + 1] + 'visualisation/plot'
    with open(plotpos.split('visualisation/')[0] + 'explplot.txt', 'w') as f:
        f.write('\n'.join(whole))

def combine2json(pos):
    rules = collections.defaultdict(lambda : [])
    for root, dirs, files in os.walk(pos):
        for file in files:
            if file.endswith('json'):

                j = os.path.join(root, file)

                blk_dup = j.split('/')[-2]

                if blk_dup not in ['blk', 'dup']:
                    continue

                rules[blk_dup].append(j)

    for blk_dup in rules:
        rules_dict = {'stats': {},
                      'preamble': {'benchmark': 'quantised',
                               'prog_arg': None,
                               'program': 'block' if blk_dup == 'blk' else 'duplicate',
                               'prog_alias': 'block' if blk_dup == 'blk' else 'duplicate'}}

        for jsonfn in rules[blk_dup]:
            dataset = jsonfn.rsplit('/')[-1].split('.csv')[0] + '.csv'

            with open(jsonfn, 'r') as f:
                f_dict = json.load(f)

            nof_rules = []
            for label in f_dict:
                for lvalue in f_dict[label]:
                    nof_rules.append(len(f_dict[label][lvalue]))

            nof_rules = sum(nof_rules)

            rules_dict['stats'][dataset] = {'status': True,
                                            'nrules': nof_rules}

        savedfn = pos.strip('/') + '/' + blk_dup + '.json'
        with open(savedfn, 'w') as f:
            json.dump(rules_dict, f, indent=4)

        yield savedfn

def blk2scater(blk_fns):

    max_nrules = set()
    for fn in blk_fns:
        max_r = set()
        with open(fn, 'r') as f:
            f_dict = json.load(f)

        for dataset in f_dict['stats']:
            nrules = f_dict['stats'][dataset]['nrules']
            max_r.add(nrules)

        max_nrules.add(max(max_r))

    max_nrules = max(max_nrules)

    pdffn = blk_fns[0][ : blk_fns[0].rfind('/') + 1] + 'scatter.pdf'

    command = 'python3 ../../../rc2/mkplot-master/mkplot.py -l -p scatter ' \
              f'-b pdf --save-to {pdffn} -k {"nrules"} --shape squared ' \
              f'-t {max_nrules} --ymax {max_nrules + 1} --ylog --ymin 1 --xlog ' \
              f'{" ".join(blk_fns)}'

    os.system(command)

def blk2table(blk_fns):


    h = ['\\begin{center}',

         '	\\centering',

         '\\begin{longtable}{|l|l|l|l|l|l|} \\hline',

        '\\multirow{2}{*}{\\textbf{Dataset}} & \\multicolumn{2}{|c|}{\\textbf{Nof rules}} '
        '& \\multirow{2}{*}{\\textbf{Dataset}} & \\multicolumn{2}{|c|}{\\textbf{Nof rules}} '
        '\\\\ \\cline{2-3} \\cline{5-6}',

         '& \\textbf{Before} & \\textbf{After} & & \\textbf{Before} & \\textbf{After} \\\\ \\hline']

    e = ['\\end{longtable}',
         ' \\end{center}]']

    fn1 = sorted(blk_fns, reverse=True)[0]
    fn2 = sorted(blk_fns, reverse=True)[1]

    with open(fn1, 'r') as f:
        f1_dict = json.load(f)

    with open(fn2, 'r') as f:
        f2_dict = json.load(f)

    q2dts = collections.defaultdict(lambda : [])

    for dataset in f2_dict['stats']:
        q = dataset.split('_')[0]
        q2dts[q].append(dataset)


    quantiles = sorted(q2dts.keys(), reverse=True)


    body = []

    dt2stats = {q: {} for q in quantiles}
    for q in quantiles:
        q2dts[q] = sorted(q2dts[q])
        for dataset in q2dts[q]:
            nrules1 = f1_dict['stats'][dataset]['nrules']
            nrules2 = f2_dict['stats'][dataset]['nrules']
            dt2stats[q][dataset] = [nrules1, nrules2]

    for q in quantiles:
        datasets = sorted(dt2stats[q].keys())
        for i in range(0, len(datasets), 2):
            dt1 = datasets[i]
            dt2 = datasets[i+1] if i + 1 <= len(datasets) - 1 else None

            dt1_nrules = dt2stats[q][dt1][:]
            dt2_nrules = dt2stats[q][dt2][:] if dt2 is not None else None

            name1 = dt1.replace('_', '\\_')
            name2 = dt2.replace('_', '\\_') if dt2 is not None else ''

            if dt2_nrules is not None:
                b = f'{name1} & ${dt1_nrules[0]}$ & ${dt1_nrules[1]}$  & '\
                    f'{name2} & ${dt2_nrules[0]}$ & ${dt2_nrules[1]}$ \\\\ \\hline'
                body.append(b)
            else:
                b = f'{name1} & ${dt1_nrules[0]}$ & ${dt1_nrules[1]}$ $ & ' \
                    f' & & \\ \\hline'
                body.append(b)
                body.append(' & & & \\ \\hline')

    whole = h + body + e

    nrulesfn = blk_fns[0][: blk_fns[0].rfind('/') + 1] + 'nrules.txt'

    with open(nrulesfn, 'w') as f:
        f.write('\n'.join(whole))

def exptime(pos, type='size'):
    logs = glob.glob(pos)
    logs = list(filter(lambda l: '_train' not in l, logs))

    logs_size = list(filter(lambda l: '_size500' in l, logs))
    logs_all = list(filter(lambda l: '_size' not in l and '_rmine' in l, logs))
    logs_none = list(filter(lambda l: '_rmine' not in l, logs))

    rdict = collections.defaultdict( lambda : collections.defaultdict(
        lambda: {"preamble": {"program": type, "prog_args": type, "prog_alias": type, "benchmark": type},
                 "stats": {}}))

    xtypes = ['abd', 'con']

    configs = {'size': logs_size, 'all': logs_all, 'none': logs_none}

    for conf in configs:
        for xtype in xtypes:
            logs = list(filter(lambda l: f'_{xtype}' in l, configs[conf]))

            for e in ["prog_alias", "prog_args", "program"]:
                rdict[conf][xtype]['preamble'][e] = f'{conf}{xtype}'

            for log in logs:
                with open(log, 'r') as f:
                    lines = f.readlines()

                sep = '_' + xtype
                dt = log.split(sep)[0].rsplit('/')[-1].replace('_data', '') + '.gz'

                nof_expls = 0
                for i in range(len(lines) - 1, -1, -1):
                    if 'all samples:' in lines[i]:
                        nof_insts = int(lines[i].split('all samples:')[-1])
                    if 'mabd nof x:' in lines[i]:
                        nof_expls += int(lines[i].split('mabd nof x:')[-1])
                    if 'mabd total time:' in lines[i]:
                        rtime = float(lines[i].split('mabd total time:')[-1])

                exprtime = round(rtime / nof_expls * 1000, 4)
                instrtime = round(rtime / nof_insts, 4)
                rdict[conf][xtype]['stats'][dt] = {'status': True,
                                                   'rtime': rtime,
                                                   'expls': nof_expls,
                                                   'exprtime': exprtime if exprtime > 0.0001 else 0.0001,
                                                   'insts': nof_insts,
                                                   'instrtime': instrtime if instrtime > 0.0001 else 0.0001}

            fdir = './rtime/exp'
            if not os.path.exists(fdir):
                os.makedirs(fdir)

            jsonfn = f'{fdir}/{conf}_{xtype}.json'

            with open(jsonfn, 'w') as f:
                json.dump(rdict[conf][xtype], f,  indent=4)

def accumrtime(type):

    exps = glob.glob('./rtime/exp/{0}*json'.format(type))
    rmine = glob.glob('./rtime/rmine/{0}*json'.format(type))[0]

    for xtype in ('abd', 'con'):
        xtype_exps = list(filter(lambda l: xtype in l, exps))
        for exp in xtype_exps:
            with open(exp, 'r') as f:
                expdict = json.load(f)
            with open(rmine, 'r') as f:
                rminedict = json.load(f)

            newdict = {"preamble": {"program": "{0}{1}rmine".format(type, xtype), "prog_args": "{0}{1}rmine".format(type, xtype),
                                    "prog_alias": "{0}{1}rmine".format(type, xtype), "benchmark": "exp"},
                       "stats": {}}

            for dt in expdict['stats']:
                exp_rtime = expdict['stats'][dt]['rtime']
                rmine_rtime = rminedict['stats'][dt]['rtime']
                sum_rtime = exp_rtime + rmine_rtime

                newdict['stats'][dt] = {'status': True, 'rtime': round(sum_rtime, 2)}

            jsonfn = './rtime/' + exp.rsplit('/')[-1].replace('.json', '') + '_rmine.json'

            with open(jsonfn, 'w') as f:
                json.dump(newdict, f, indent=4)

if __name__ == '__main__':
    options = Options(sys.argv)

    """

        Summarise runtime

    """
    exptime('./exp/*.txt')

    #for type in ['all', 'size']:
    #    accumrtime(type)

    print(aa)

    """
    
    To latex
    
    """

    # clear; python3 stat.py -i ./compr/explstat.json ./compr/explcmpr.json
    # compare configurations with the stat not applying background knowledge
    # dict = {name: xtype: sizeconf: configs: stat
    with open(options.files[0], 'r') as f:
        cmpr_dict = json.load(f)

    q2dt = collections.defaultdict(lambda : [])
    for dt in cmpr_dict:
        if 'meteo' in dt:
            continue
        q = dt.split('_')[0].strip()
        if q == 'q3':
            continue
        q2dt[q].append(dt)

    for q in q2dt:
        q2dt[q].sort()

    # apply all rules
    allrules(cmpr_dict, q2dt)
    # compare <= size limit
    sizestop(cmpr_dict, q2dt)


    """
    
    To generate json and then visualise
    
    """

    # combine the number of min_size, nof_expls for each instance and the save to json
    tojson(options.dtinfo)
    # generate scatter plots
    scatter(options.dtinfo[ : options.dtinfo.rfind('/') + 1] + 'visualisation')

    """
    latex
    """

    # scatter plots of min_size, nof_expls to latex
    sct2latex(options.model[: options.model.rfind('/') + 1] + 'visualisation/plot')

    '''
    """
    
    The number of rules 
    with/ without blocking
    
    """
    # ./background
    # clear; python3 stat.py -D ./background/
    blk_fns= combine2json(options.dataset)
    blk_fns = list(blk_fns)
    
    # generate scatter
    #blk2scater(blk_fns)
    # generate comparison tables in latex
    blk2table(blk_fns)
    '''