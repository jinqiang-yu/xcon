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
    stats_dict = collections.defaultdict(lambda :
                                         {"preamble": {"program": None, "prog_args": None, "prog_alias": None, "benchmark": None},
                                          "stats": collections.defaultdict(lambda: {'status': True })})


    for log in logs:
        dtname = log.rsplit('/')[-1].split('_', maxsplit=2)[-1].split('_test1')[0]
        for e in ['program', 'prog_args', 'prog_alias']:
            stats_dict[dtname]['preamble'][e] = "{0}$_{1}$".format(appr, '{' + model + '}')
        with open(log, 'r') as f:
            lines = f.readlines()
        lines = list(filter(lambda l: len(l.strip()) > 0, lines))
        lines = list(map(lambda l: l.strip(), lines))

        inst_mark = 'explaining:'
        if appr == 'lime':
            expl_mark = 'Features in explanation:'
        elif appr == 'shap':
            expl_mark = 'Features in explanation:'
        elif appr == 'anchor':
            expl_mark = 'Anchor:'
        else:
            #print('approach wrong')
            exit(1)

        insts = []

        for i, line in enumerate(lines):
            if inst_mark in line:
                insts.append(i)

        insts.append(len(lines))

        for i, start in enumerate(insts[:-1]):
            end = insts[i + 1]
            inst = lines[insts[i]].split(inst_mark)[-1].strip()
            inst_list = inst.split(' AND ')
            inst_list[0] = inst_list[0].split(maxsplit=1)[-1].strip()
            inst_list[-1] = inst_list[-1].rsplit(' THEN ')[0].strip()
            inst_set = set(inst_list)
            dt_inst = f'{dtname}_{inst}'
            pred = inst.rsplit('THEN')[-1].split('=', maxsplit=1)[-1]. strip()

            for ii in range(start + 1, end):
                if expl_mark in lines[ii]:
                    if appr == 'anchor':
                        expl = lines[ii].split(expl_mark)[-1].strip()
                        if len(expl) > 0:
                            expl_ = expl.split(' AND ')
                            expl = []
                            for x in expl_:
                                for fv in inst_set:
                                    if x in fv:
                                        expl.append(fv.split(' = ', maxsplit=1)[0])
                                        break
                                else:
                                    print('wrong expl')
                                    exit(1)
                        else:
                            expl = []
                    else:
                        classes = lines[ii-1].split(':', maxsplit=1)[-1].split(', ')
                        classes = list(map(lambda l: l.strip().strip('[').strip(']').strip().strip("'"), classes))
                        expl = lines[ii].split(expl_mark)[-1].strip().strip('[').strip(']').split("), ('")
                        expl = list(map(lambda l: l.split("', "), expl))
                        pos_feats = []
                        neg_feats = []
                        for e in expl:
                            f = e[0].replace("('", "").strip()
                            imprt = e[1].strip(')')
                            imprt = float(imprt)
                            if imprt > 0:
                                pos_feats.append(f)
                            elif imprt < 0:
                                neg_feats.append(f)

                        if len(classes) == 2 and classes.index(pred) == 0:
                            expl = neg_feats
                        else:
                            expl = pos_feats

                if 'time: ' in lines[ii]:
                    rtime = float(lines[ii].split('time:', maxsplit=1)[-1])
                    break

            stats_dict[dtname]['stats'][inst]['expl'] = expl
            stats_dict[dtname]['stats'][inst]['minsize'] = len(expl)
            stats_dict[dtname]['stats'][inst]['exprtime'] = rtime * 1000

    for dtname in stats_dict:
        saved_dir = f'../stats/hexp_expls/{appr}/{model}'

        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)

        jsonfn = f'{saved_dir}/{dtname}.json'
        with open(jsonfn, 'w') as f:
            json.dump(stats_dict[dtname], f, indent=4)

def cmb2isnt(files, appr, model):

    stats_dict  = {"preamble": {"program": None, "prog_args": None, "prog_alias": None, "benchmark": None},
                   "stats": collections.defaultdict(lambda: {'status': True })}

    for e in ['program', 'prog_args', 'prog_alias']:
        stats_dict['preamble'][e] = "{0}$_{1}$".format(appr, '{' + model + '}')

    for file in files:
        with open(file, 'r') as f:
            info = json.load(f)
        dtname = file.rsplit('/')[-1].rsplit('.json')[0]

        for inst in info['stats']:
            dtname_inst = '{0}_{1}'.format(dtname, inst)
            exprtime = info['stats'][inst]['exprtime']
            stats_dict['stats'][dtname_inst]['exprtime'] = exprtime

    saved_dir = '../stats/rtime/hexp'

    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    saved_file = '{0}/{1}_{2}.json'.format(saved_dir, appr, model)

    with open(saved_file, 'w') as f:
        json.dump(stats_dict, f, indent=4)

    return saved_file

def rtimecactus(files, key, model, x_label):

    maxkey = 0
    minkey = 0

    for file in files:
        with open(file, 'r') as f:
            info = json.load(f)

        exprtimes = [info['stats'][dt][key] for dt in info['stats']]

        cur_maxk = max(exprtimes)
        cur_mink = min(exprtimes)

        if cur_maxk > maxkey:
            maxkey = cur_maxk
        if cur_mink < minkey:
            minkey = cur_mink

    saved_dir = '../plots/hexp/rtime'

    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    if 'instance' in x_label .lower():
        pdf_name = '{0}/cactus_{1}_inst_rtime.pdf'.format(saved_dir, model)
        cmd = 'python ./gnrt_plots/mkplot/mkplot.py --shape squared --font-sz 16 -w 1 --ms 5 --sort ' \
              '-l --legend prog_alias --xlabel {0} --ylabel "Runtime(ms)" -t {1} ' \
              '--lloc "upper left" --ymin {2} -b pdf ' \
              '--save-to {3} -k {4} {5}'.format(x_label,
                                                maxkey,
                                                minkey,
                                                pdf_name,
                                                key,
                                                ' '.join(files))
    else:
        pdf_name = '{0}/cactus_{1}_dt_rtime.pdf'.format(saved_dir, model)
        cmd = 'python ./gnrt_plots/mkplot/mkplot.py --shape squared --font-sz 18 -w 1.5 --ms 9 --sort ' \
              '-l --legend prog_alias --xlabel {0} --ylabel "Runtime(ms)" -t {1} ' \
              '--lloc "upper left" --ymin {2} -b pdf ' \
              '--save-to {3} -k {4} {5}'.format(x_label,
                                                maxkey,
                                                minkey,
                                                pdf_name,
                                                key,
                                                ' '.join(files))

    os.system(cmd)

def rtime_dataset(file, model, bg):

    with open(file, 'r') as f:
        info = json.load(f)

    stats_dict = {"preamble": {"program": None, "prog_args": None, "prog_alias": None, "benchmark": None},
                  "stats": collections.defaultdict(lambda: {'status': True, 'exprtime': []})}

    for e in ['program', 'prog_args', 'prog_alias']:
        stats_dict['preamble'][e] = info['preamble'][e].replace('{' + m + '}', '{' + m + ',' + xtype_ + '}')

    for dt_inst in info['stats']:
        dt = dt_inst[:dt_inst.find('_IF')]
        inst = dt_inst[dt_inst.find('_IF') + 1 :]
        exprtime = info['stats'][dt_inst]['exprtime']
        stats_dict['stats'][dt]['exprtime'].append(exprtime)

    for dt in stats_dict['stats']:
        stats_dict['stats'][dt]['exprtime'] = statistics.mean(stats_dict['stats'][dt]['exprtime'])

    saved_dir = '../stats/rtime/hexp'
    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    saved_file = '{0}/dt_formal_{1}_{2}_{3}.json'.format(saved_dir, model, 'axp' if 'abd' in file else 'cxp', bg)

    with open(saved_file, 'w') as f:
        json.dump(stats_dict, f, indent=4)

    return saved_file

def cmb2dataset(files, appr, model):

    stats_dict  = {"preamble": {"program": None, "prog_args": None, "prog_alias": None, "benchmark": None},
                   "stats": collections.defaultdict(lambda: {'status': True, })}

    for e in ['program', 'prog_args', 'prog_alias']:
        stats_dict['preamble'][e] = "{0}$_{1}$".format(appr, '{' + model + '}')

    for file in files:
        with open(file, 'r') as f:
            info = json.load(f)
        dtname = file.rsplit('/')[-1].rsplit('.json')[0]
        exprtimes = []
        for inst in info['stats']:
            exprtime = info['stats'][inst]['exprtime']
            exprtimes.append(exprtime)

        stats_dict['stats'][dtname]['exprtime'] = statistics.mean(exprtimes)

    saved_dir = '../stats/rtime/hexp'

    if not os.path.isdir(saved_dir):
        os.makedirs(saved_dir)

    saved_file = '{0}/dt_{1}_{2}.json'.format(saved_dir, appr, model)

    with open(saved_file, 'w') as f:
        json.dump(stats_dict, f, indent=4)

    return saved_file

if __name__ == '__main__':

    apprmod2logs = collections.defaultdict(lambda : [])

    for root, dirs, files in os.walk('../logs/hexp'):
        for file in files:
            if file.endswith('.log'):
                dt = file.split('_', maxsplit=2)[-1].rsplit('_test1')[0]
                model, appr = file.split('_')[:2]
                apprmod2logs[(appr, model)].append(os.path.join(root, file))

    for appr, model in apprmod2logs:
        logs = apprmod2logs[tuple([appr, model])]
        extract_exps(logs, model, appr)


    """
    
    Runtime
    
    """
    models = ['dl', 'bt', 'bnn']
    #../stats/hexp_expls each file for one dataset
    #

    # runtime of an instance
    m2expls = {}
    for m in models:
        m2expls[m] = []
        for xtype in ['abd', 'con']:
            for conf in ['ori', 'size5']:
                file = '../stats/expls/{0}/{1}_{2}.json'.format(m, conf, xtype)
                with open(file, 'r') as f:
                    info = json.load(f)
                xtype_ = 'axp' if xtype == 'abd' else 'cxp'
                for e in ["program", "prog_args", "prog_alias"]:
                    info['preamble'][e] = info['preamble'][e].replace('{' + m + '}', '{' + m + ',' + xtype_ + '}')

                saved_dir = '../stats/rtime/hexp'
                if not os.path.isdir(saved_dir):
                    os.makedirs(saved_dir)
                new_file = '{0}/formal_{1}_{2}_{3}.json'.format(saved_dir, m, xtype_, 'bg' if conf!= 'ori' else 'nobg')

                with open(new_file, 'w') as f:
                    json.dump(info, f, indent=4)
                m2expls[m].append(new_file)

    apprmod2files = collections.defaultdict(lambda : [])
    for root, dirs, files in os.walk('../stats/hexp_expls'):
        for file in files:
            if file.endswith('.json'):
                appr, model = root.strip('/').split('/')[-2:]
                apprmod2files[tuple([appr, model])].append(os.path.join(root, file))

    m2inst_files = {m: m2expls[m][:] for m in models}

    for appr, model in apprmod2files:
        files = apprmod2files[tuple([appr, model])]
        m2inst_files[model].append(cmb2isnt(files, appr, model))

    key = 'exprtime'
    for model in m2inst_files:
        inst_files = m2inst_files[model]
        rtimecactus(inst_files, key, model, 'Instances')

    # runtime/instance of a dataset
    m2expls = {}
    for m in models:
        m2expls[m] = []
        for xtype in ['abd', 'con']:
            for conf in ['ori', 'size5']:
                file = '../stats/expls/{0}/{1}_{2}.json'.format(m, conf, xtype)
                m2expls[m].append(rtime_dataset(file, m, 'bg' if conf != 'ori' else 'nobg'))

    apprmod2files = collections.defaultdict(lambda: [])
    for root, dirs, files in os.walk('../stats/hexp_expls'):
        for file in files:
            if file.endswith('.json'):
                appr, model = root.strip('/').split('/')[-2:]
                apprmod2files[tuple([appr, model])].append(os.path.join(root, file))

    m2dt_files = {m: m2expls[m][:] for m in models}

    for appr, model in apprmod2files:
        files = apprmod2files[tuple([appr, model])]
        m2dt_files[model].append(cmb2dataset(files, appr, model))

    key = 'exprtime'
    for model in m2inst_files:
        inst_files = m2dt_files[model]
        rtimecactus(inst_files, key, model, 'Datasets')
    #inst_files =
    #minsize
    #exprtime

    exit()
