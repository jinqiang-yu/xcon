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

def rextractrtime(logs, appr):
    timelimit = 3600 * 24
    logs_size = list(filter(lambda l: '_size5' in l, logs))
    logs_all = list(filter(lambda l: '_all' in l, logs))

    rdict = collections.defaultdict(lambda : { "preamble": {"program": None, "prog_args": None, "prog_alias": None, "benchmark": None},
              "stats": {}})

    avg_rdict = collections.defaultdict(lambda : { "preamble": {"program": None, "prog_args": None, "prog_alias": None, "benchmark": None},
              "stats": collections.defaultdict(lambda : {'status': True,
                                                          'rtime': []})})

    configs = {'5': logs_size, 'all': logs_all}
    configs = {'5': logs_size}

    #conf2label = {'size': 'apriori$_5$', 'all': 'apriori$_{all}$'}
    conf2label = {'5': appr, 'all': appr}

    for conf in configs:
        logs = configs[conf]

        for e in ["prog_alias", "prog_args", "program"]:

            rdict[conf]['preamble'][e] = conf2label[conf]
            avg_rdict[conf]['preamble'][e] = conf2label[conf]

        for log in logs:
            with open(log, 'r') as f:
                lines = f.readlines()

            dt = log.rsplit('rextract_')[-1].rsplit('.csv_')[0]
            dtname = dt[:dt.rfind('_')]

            for i in range(len(lines)-1, -1, -1):
                if 'c3 total time:' in lines[i]:
                    status = True
                    rtime = float(lines[i].split('c3 total time:')[-1])
                    break
            else:
                status = False
                rtime = timelimit

            rdict[conf]['stats'][dt] = {'status': status, 'rtime': rtime}
            avg_rdict[conf]['stats'][dtname]['rtime'].append(rtime)

        fdir = '../stats/{0}rtime/rextract'.format(f'{appr}_' if appr != 'xcon' else '')
        if not os.path.exists(fdir):
                os.makedirs(fdir)

        jsonfn = f'{fdir}/{conf}.json'

        with open(jsonfn, 'w') as f:
            json.dump(rdict[conf], f, indent=4)

        for dtname in avg_rdict[conf]['stats']:
            assert len(avg_rdict[conf]['stats'][dtname]['rtime']) == 5
            avg_rdict[conf]['stats'][dtname]['rtime'] = statistics.mean(avg_rdict[conf]['stats'][dtname]['rtime'])
            if avg_rdict[conf]['stats'][dtname]['rtime'] == timelimit:
                avg_rdict[conf]['stats'][dtname]['status'] = False

        avg_jsonfn = f'{fdir}/avg_{conf}.json'
        with open(avg_jsonfn, 'w') as f:
            json.dump(avg_rdict[conf], f, indent=4)

def rtimescatter(files, appr):
    keys = ['rtime']
    for key in keys:
        maxkey = cmptmax(files, key)
        maxkey = maxkey if maxkey >= 1 else 1

        fdir = '../plots/{0}/'.format(appr)
        if not os.path.exists(fdir):
                os.makedirs(fdir)

        pdffn =  fdir + files[-1].rsplit('/')[-1].replace('.json', '') + f'_{key}_scatter.pdf'
        cmd = 'python ./gnrt_plots/mkplot/mkplot.py ' \
              '-l -p scatter -b pdf --save-to {0} -k {1} --shape squared --font-sz 20 --tol-loc none ' \
              '-t {2} --ylog --ymin 0.1 --xlog {3}'.format(pdffn, key,
                                                                       maxkey, ' '.join(files))
        #' --ymax {2} '
        #print('\ncmd:\n{0}'.format(cmd))
        os.system(cmd)
        #print()

def rtimecactus(files, avg=False):
    keys = ['rtime']

    for key in keys:
        maxkey = cmptmax(files, key)

        fdir = '../plots/apriori/'
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        pdffn = fdir + files[-1].rsplit('/')[-1].replace('.json', '') + f'_{key}_cactus.pdf'

        xlabel = 'Dataset'
        cmd = 'python ./gnrt_plots/mkplot/mkplot.py --font-sz 22 ' \
              '-w 12 -l --legend prog_alias --xlabel {0} -t {1} -b pdf ' \
              '--save-to {2} -k {3} {4}'.format(xlabel, maxkey, pdffn, key, ' '.join(files))
        #print(pdffn)
        #print('\ncmd:', cmd)
        os.system(cmd)
        #print()

def count_rules(info):
    rules = 0
    for feat in info:
        for fv in info[feat]:
            rules += len(info[feat][fv])
    return rules

def get_nof_rules(logs, appr):
    rdict = collections.defaultdict(
        lambda: {"preamble": {"program": None, "prog_args": None, "prog_alias": None, "benchmark": None},
                 "stats": {}})

    avg_rdict = collections.defaultdict(
        lambda: {"preamble": {"program": None, "prog_args": None, "prog_alias": None, "benchmark": None},
                 "stats": collections.defaultdict(lambda: {'status': True,
                                                           'rules': []})})

    for log in logs:
        with open(log, 'r') as f:
            lines = f.readlines()

        conf = '5' if '_size5' in log else 'all'
        dt = log.rsplit('rextract_')[-1].rsplit('.csv_')[0]
        dtname = dt[:dt.rfind('_')]

        for i in range(len(lines)-1, -1, -1):
            line = lines[i]
            if 'Rules are saved to' in line:
                status = True
                json_file = line.split('Rules are saved to')[-1].strip()
                assert os.path.isfile(json_file)
                with open(json_file, 'r') as f:
                    info = json.load(f)
                nof_rules = count_rules(info)
                break
        else:
            status = False
            nof_rules = 0

        rdict[conf]['stats'][dt] = {'status': status, 'rules': nof_rules}
        avg_rdict[conf]['stats'][dtname]['rules'].append(nof_rules)

    for conf in rdict:

        for e in ["prog_alias", "prog_args", "program"]:
            #n = '5' if conf != 'all' else conf
            #rdict[conf]['preamble'][e] = 'apriori$_{0}$'.format('{' + n + '}') if 'apriori_' in logs[0] else 'xcon$_{0}$'.format('{' + n + '}')
            #avg_rdict[conf]['preamble'][e] = 'apriori$_{0}$'.format('{' + n + '}') if 'apriori_' in logs[0] else 'xcon$_{0}$'.format('{' + n + '}')

            rdict[conf]['preamble'][e] = appr
            avg_rdict[conf]['preamble'][e] = appr

        fdir = '../stats/{0}nof_rules'.format(f'{appr}_' if f'{appr}_' in logs[0] else '')
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        jsonfn = f'{fdir}/{conf}.json'

        with open(jsonfn, 'w') as f:
            json.dump(rdict[conf], f, indent=4)

        for dtname in avg_rdict[conf]['stats']:
            assert len(avg_rdict[conf]['stats'][dtname]['rules']) == 5
            if any(rules==0 for rules in avg_rdict[conf]['stats'][dtname]['rules']):
                avg_rdict[conf]['stats'][dtname]['status'] = False
            avg_rdict[conf]['stats'][dtname]['rules'] = statistics.mean(avg_rdict[conf]['stats'][dtname]['rules'])

        avg_jsonfn = f'{fdir}/avg_{conf}.json'
        with open(avg_jsonfn, 'w') as f:
            json.dump(avg_rdict[conf], f, indent=4)

def rulescatter(files):
    keys = ['rules']
    for key in keys:
        maxkey = cmptmax(files, key)
        maxkey = maxkey if maxkey >= 1 else 1

        fdir = '../plots/eclat/'
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        pdffn = fdir + files[-1].rsplit('/')[-1].replace('.json', '') + f'_{key}_scatter.pdf'

        cmd = 'python ./gnrt_plots/mkplot/mkplot.py ' \
              '-l -p scatter -b pdf --save-to {0} -k {1} --shape squared --font-sz 20 --tol-loc none ' \
              '-t {2} --ylog --ymax {2} --ymin 0.1 --xlog {3}'.format(pdffn, key,
                                                                       maxkey, ' '.join(files))
        #print('\ncmd:\n{0}'.format(cmd))
        os.system(cmd)
        #print()

def rulescactus(files):
    keys = ['rules']

    for key in keys:
        maxkey = cmptmax(files, key)

        fdir = '../plots/apriori/'
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        pdffn = fdir + files[-1].rsplit('/')[-1].replace('.json', '') + f'_{key}_cactus.pdf'

        xlabel = 'Dataset'
        cmd = 'python ./gnrt_plots/mkplot/mkplot.py --font-sz 22 --tol-loc none ' \
              '-w 12 -l --legend prog_alias --xlabel {0} -t {1} -b pdf ' \
              '--save-to {2} -k {3} {4}'.format(xlabel, maxkey, pdffn, key, ' '.join(files))
        #print(pdffn)
        #print('\ncmd:', cmd)
        os.system(cmd)
        #print()

if __name__ == '__main__':

    #appr = sys.argv[1]

    """
    
    Runtime
    
    """

    #pasrse logs
    for appr in ['apriori', 'xcon', 'eclat']:
        logs = glob.glob('../{0}logs/rextract/*.log'.format(f'{appr}_' if appr != 'xcon' else ''))
        rextractrtime(logs, appr)

    # generate plots
    '''
    a_avg2files = {}
    
    a_avg2files[False] = ['../stats/apriori_rtime/rextract/all.json',
                          '../stats/apriori_rtime/rextract/size.json']

    a_avg2files[True] = ['../stats/apriori_rtime/rextract/avg_all.json',
                       '../stats/apriori_rtime/rextract/avg_size.json']


    ori_avg2files = {}
    ori_avg2files[False] = ['../stats/rtime/rextract/all.json',
                        '../stats/rtime/rextract/size.json']

    ori_avg2files[True] = ['../stats/rtime/rextract/avg_all.json',
                       '../stats/rtime/rextract/avg_size.json']
    '''

    a_avg2files = {}
    a_avg2files[False] = ['../stats/apriori_rtime/rextract/5.json']
    a_avg2files[True] = ['../stats/apriori_rtime/rextract/avg_5.json']

    e_avg2files = {}
    e_avg2files[False] = ['../stats/eclat_rtime/rextract/5.json']
    e_avg2files[True] = ['../stats/eclat_rtime/rextract/avg_5.json']

    ori_avg2files = {}
    ori_avg2files[False] = ['../stats/rtime/rextract/5.json']
    ori_avg2files[True] = ['../stats/rtime/rextract/avg_5.json']

    for avg in [True]:
        a_files = a_avg2files[avg]
        e_files = e_avg2files[avg]
        ori_files = ori_avg2files[avg]

        rtimescatter([ori_files[0], a_files[0]], appr='apriori')
        rtimescatter([ori_files[0], e_files[0]], appr='eclat')
        pass

    """
    
    nof_rules
    
    """
    # parse
    logs = glob.glob('../apriori_logs/rextract/*.log')
    #get_nof_rules(logs, 'apriori')

    logs = glob.glob('../eclat_logs/rextract/*.log')
    #get_nof_rules(logs, 'eclat')

    logs = glob.glob('../logs/rextract/*.log')
    #get_nof_rules(logs, 'xcon')

    # plots
    #apriori_nof_rules = glob.glob('../stats/apriori_nof_rules/*.json')
    eclat_nof_rules = glob.glob('../stats/eclat_nof_rules/*.json')

    for e_name in eclat_nof_rules:

        # apriori info
        with open(e_name, 'r') as f:
            eclat_info = json.load(f)

        # xcon info
        name = e_name.replace('eclat_nof_rules/', 'nof_rules/')
        with open(name, 'r') as f:
            info = json.load(f)

        dtnames = list(eclat_info['stats'].keys())
        for dt in dtnames:
            apriori_status = eclat_info['stats'][dt]['status']
            if not apriori_status:
                del eclat_info['stats'][dt]
                del info['stats'][dt]

        assert len(eclat_info['stats'].keys()) == len(info['stats'])

        e_new_dir = e_name[: e_name.rfind('/')] + '/allsolved'
        new_dir = name[: name.rfind('/')] + '/allsolved'

        if not os.path.exists(e_new_dir):
            os.makedirs(e_new_dir)

        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        with open(f'{e_new_dir}/{e_name.rsplit("/")[-1]}', 'w') as f:
            json.dump(eclat_info, f, indent=4)

        with open(f'{new_dir}/{name.rsplit("/")[-1]}', 'w') as f:
            json.dump(info, f, indent=4)


    # generate plots
    '''
    a_avg2files = {}
    a_avg2files[False] = ['../stats/apriori_nof_rules/allsolved/all.json',
                        '../stats/apriori_nof_rules/allsolved/size.json']
    a_avg2files[True] = ['../stats/apriori_nof_rules/allsolved/avg_all.json',
                       '../stats/apriori_nof_rules/allsolved/avg_size.json']

    ori_avg2files = {}
    ori_avg2files[False] = ['../stats/nof_rules/allsolved/all.json',
                            '../stats/nof_rules/allsolved/size.json']
    ori_avg2files[True] = ['../stats/nof_rules/allsolved/avg_all.json',
                           '../stats/nof_rules/allsolved/avg_size.json']
    '''

    a_avg2files = {}
    a_avg2files[False] = ['../stats/apriori_nof_rules/allsolved/5.json']
    a_avg2files[True] = ['../stats/apriori_nof_rules/allsolved/avg_5.json']

    e_avg2files = {}
    e_avg2files[False] = ['../stats/eclat_nof_rules/allsolved/5.json']
    e_avg2files[True] = ['../stats/eclat_nof_rules/allsolved/avg_5.json']

    ori_avg2files = {}
    ori_avg2files[False] = ['../stats/nof_rules/allsolved/5.json']
    ori_avg2files[True] = ['../stats/nof_rules/allsolved/avg_5.json']

    #for avg in [False, True]:
    for avg in [True]:
        files = e_avg2files[avg]
        ori_files = ori_avg2files[avg]

        rulescatter([ori_files[0], files[0]])
        #rulescactus([ori_files[0], files[0]])
        pass
    exit()