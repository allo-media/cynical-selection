#!/usr/bin/env python3

import argparse
import math
import bisect
import re
import logging
import time
from decimal import localcontext


parser = argparse.ArgumentParser(description='Allomedia data selection tool')
parser.add_argument('--task', required=True, type=str,
                    help='Task file, the sentences we want to mimic')
parser.add_argument('--unadapted', required=True, type=str,
                    help='Unadapted senteces, the pool we want to pick from')
parser.add_argument('--batch', action='store_true', dest='batch',
                    help='Set this flag to enable batch mode')
parser.add_argument('--no-batch', action='store_false', dest='batch',
                    help='Set this flag to disable batch mode')
parser.set_defaults(batch=False)
parser.add_argument('--mincount', type=int, default=3,
                    help='Minimum wordcount in each file')
parser.add_argument('--keep', action='store_true', dest='keep',
                    help='Set this flag to keep boring words')
parser.add_argument('--no-keep', action='store_false', dest='keep',
                    help='Set this flag to NOT keep boring words')
parser.set_defaults(keep=False)
parser.add_argument('--lower', action='store_true', dest='lower',
                    help='Set this flag to lowercase all text')
parser.add_argument('--no-lower', action='store_false', dest='lower',
                    help='Set this flag to NOT lowercase all text')
parser.set_defaults(lower=True)
parser.add_argument('--maxlen', type=int, default=250,
                    help='Maximum sentence length')
parser.add_argument('--smoothing', type=float, default=0.01,
                    help='Smoothing factor for lenght penalty computation')


def compute_counts(corpus):
    words = {}
    size = 0

    for line in corpus:
        for token in line.split():
            if token in words:
                words[token]['count'] += 1
            else:
                words[token] = {}
                words[token]['count'] = 1
            size += 1

    # for item in sorted(words.items(), key=lambda x: (-x[1], x[0])):
    for k in words.keys():
        words[k]['prob'] = words[k]['count'] / size

    return words


def float_to_str(num):
    with localcontext() as ctx:
        ctx.prec = 20
        d = ctx.create_decimal(repr(num))
    return format(d, 'f')


def compute_ratios(task_vocab, unadapted_vocab):
    ratios = {}
    sizes = {}
    sizes['task'] = 0
    sizes['unadapted'] = 0

    for word in unadapted_vocab.keys():
        ratios[word] = {}
        if word in task_vocab:
            ratios[word]['delta'] = (task_vocab[word]['prob'] /
                                     unadapted_vocab[word]['prob'])
            ratios[word]['t_prob'] = task_vocab[word]['prob']
            ratios[word]['t_count'] = task_vocab[word]['count']
            sizes['task'] += task_vocab[word]['count']
            del task_vocab[word]
        else:
            ratios[word]['delta'] = 0.5 / unadapted_vocab[word]['count']
            ratios[word]['t_prob'] = 0
            ratios[word]['t_count'] = 0
        ratios[word]['u_prob'] = unadapted_vocab[word]['prob']
        ratios[word]['u_count'] = unadapted_vocab[word]['count']
        sizes['unadapted'] += unadapted_vocab[word]['count']

    for word in task_vocab.keys():
        ratios[word] = {}
        ratios[word]['delta'] = task_vocab[word]['count'] * 2
        ratios[word]['t_prob'] = task_vocab[word]['prob']
        ratios[word]['t_count'] = task_vocab[word]['count']
        ratios[word]['u_prob'] = 0
        ratios[word]['u_count'] = 0
        sizes['task'] += task_vocab[word]['count']

    return ratios, sizes


def squish_ratios(ratios, mincount, keep):
    replace = {}
    re_band = re.compile(r'.*\.(?P<band>0*).*')
    ratios['@@IMPOSSIBLE'] = {}
    ratios['@@USELESS'] = {}
    ratios['@@DUBIOUS'] = {}
    ratios['@@IMPOSSIBLE']['t_count'] = 0
    ratios['@@USELESS']['t_count'] = 0
    ratios['@@DUBIOUS']['t_count'] = 0
    ratios['@@IMPOSSIBLE']['u_count'] = 0
    ratios['@@USELESS']['u_count'] = 0
    ratios['@@DUBIOUS']['u_count'] = 0

    for word in list(ratios.keys()):
        if word.startswith('@@'):
            continue
        if ratios[word]['t_count'] == 0:
            replace[word] = '@@USELESS'
            ratios['@@USELESS']['u_count'] += ratios[word]['u_count']
            del ratios[word]
        elif ratios[word]['u_count'] == 0:
            replace[word] = '@@IMPOSSIBLE'
            ratios['@@IMPOSSIBLE']['t_count'] += ratios[word]['t_count']
            del ratios[word]
        elif ratios[word]['t_count'] < mincount \
                and ratios[word]['u_count'] < mincount:
            replace[word] = '@@DUBIOUS'
            ratios['@@DUBIOUS']['t_count'] += ratios[word]['t_count']
            ratios['@@DUBIOUS']['u_count'] += ratios[word]['u_count']
            del ratios[word]
        elif ratios[word]['delta'] < math.exp(1) \
                and ratios[word]['delta'] > math.exp(-1) \
                and not keep:
            band = re_band.sub(r'\g<band>',
                               float_to_str(ratios[word]['u_prob']))
            bucket = '@@BORING__' + band
            replace[word] = bucket
            if bucket not in ratios:
                ratios[bucket] = {}
                ratios[bucket]['t_count'] = 0
                ratios[bucket]['u_count'] = 0
            ratios[bucket]['delta'] = bucket
            ratios[bucket]['t_count'] += ratios[word]['t_count']
            ratios[bucket]['u_count'] += ratios[word]['u_count']
            del ratios[word]
        elif ratios[word]['delta'] < 1:
            trunc = int(math.log(ratios[word]['delta']))
            bucket = '@@' + str(trunc)
            replace[word] = bucket
            if bucket not in ratios:
                ratios[bucket] = {}
                ratios[bucket]['t_count'] = 0
                ratios[bucket]['u_count'] = 0
            ratios[bucket]['delta'] = bucket
            ratios[bucket]['t_count'] += ratios[word]['t_count']
            ratios[bucket]['u_count'] += ratios[word]['u_count']
            del ratios[word]
        else:
            replace[word] = word

    return ratios, replace


def init_model(ratios, sizes, currmodel):
    for word in ratios.keys():
        ratios[word]['hconstant'] = ratios[word]['t_count'] / sizes['task']
        if word not in currmodel:
            currmodel[word] = {}
            currmodel[word]['prob'] = 0
            currmodel[word]['count'] = 0
    return ratios, currmodel


def squish_corpus(corpus, replace):
    squished = []
    for line in corpus:
        tokens = line.split()
        for i, token in enumerate(tokens):
            tokens[i] = replace[token]
        squished.append(' '.join(tokens))
    return squished


def init_penalty(maxlen, smoothing):
    penalty = {}
    for i in range(maxlen):
        penalty[i] = math.log((i + 2 * smoothing) / smoothing)
    return penalty


def count_tokens(tokens):
    count = {}
    for token in tokens:
        if token in count:
            count[token] += 1
        else:
            count[token] = 1
    return count


def index_unadapted(squish, ratios, smoothing, penalty, logger):
    pool = {}

    for lineid, line in enumerate(squish):
        tokens = line.split()
        if len(tokens) > len(penalty):
            continue
        pool[lineid] = {}
        pool[lineid]['string'] = line
        pool[lineid]['count'] = len(tokens)

        sge = 0
        count = count_tokens(tokens)
        for token in count.keys():
            sge += (ratios[token]['hconstant'] * math.log(smoothing /
                                                          count[token]))
            score = penalty[len(tokens)] + sge
            if 'line_list' not in ratios[token]:
                ratios[token]['line_list'] = []
            ratios[token]['line_list'].append((score, sge, lineid))
        pool[lineid]['SGE'] = sge

    for word in list(ratios.keys()):
        if 'line_list' in ratios[word]:
            ratios[word]['line_list'] = sorted(ratios[word]['line_list'],
                                               key=lambda x: x[0])
        else:
            del ratios[word]
            logger.debug('deleted {}, no lines'.format(word))

    return pool, ratios


def init_estimates(ratios, smoothing):
    wge = []
    re_band = re.compile(r'^@@BORING__0{1,7}')

    for token in ratios.keys():
        ratios[token]['WGE'] = ratios[token]['hconstant'] * \
                math.log(smoothing / 1)
        if token == '@@DUBIOUS' \
           or token == '@@USELESS' \
           or token == '@@IMPOSSIBLE' \
           or token.startswith('@@-') \
           or re_band.match(token) is not None:
            continue
        wge.append((ratios[token]['WGE'], token, ratios[token]['hconstant']))

    wge = sorted(wge, key=lambda x: x[0])

    return wge


def main_loop(selected, pool, ratios, currmodel,
              penalty, wge, smoothing, batch, logger):
    maxiter = len(pool)
    linecount = 1
    currmodel_linecount = 0
    currmodel_wordcount = 0
    currmodel_score = 0
    logger.debug('Running for maximum {} iterations'.format(maxiter))
    while(maxiter > 0):
        if not wge:
            logger.debug('wge is empty: {}'.format(wge))
            logger.debug('words left: {}'.format(ratios.keys()))
            logger.debug("I think we're done here.")
            break
        (WGE, bestword, hconstant) = wge[0]
        logger.debug('best word: {}'.format(bestword))
        logger.debug('(WGE={} hconstant={})'.format(WGE, hconstant))
        n_lines_for_bestword = len(ratios[bestword]['line_list'])
        if n_lines_for_bestword == 0:
            del ratios[bestword]
            logger.debug(
                'No lines left for word {}, deleting it. {} words left'.format(
                    bestword, len(ratios)))
            wordindex = [i for i, tup in enumerate(wge) if tup[1] == bestword]
            del wge[wordindex[0]:wordindex[0] + 1]
            continue

        first_bestword_id = 0
        indices_to_prune = []
        score_threshold = 0
        for i in range(n_lines_for_bestword):
            first_bestword_id = ratios[bestword]['line_list'][i][2]
            if first_bestword_id in pool:
                tokens = pool[first_bestword_id]['string'].split()
                sge = 0
                count = count_tokens(tokens)
                for token in count.keys():
                    sge += (ratios[token]['hconstant'] *
                            math.log((currmodel[token]['count'] + smoothing) /
                                     (currmodel[token]['count'] +
                                      count[token])))
                pool[first_bestword_id]['SGE'] = sge
                score_threshold = penalty[len(tokens)] + sge
                logger.debug('SGE: {} => {}'.format(
                    ratios[bestword]['line_list'][i][1], sge))
                logger.debug('score: {} => {}'.format(
                    ratios[bestword]['line_list'][i][0], score_threshold))
                break
            else:
                indices_to_prune.append(i)

        while indices_to_prune:
            i = indices_to_prune.pop()
            del ratios[bestword]['line_list'][i]
            logger.debug('Pruned the ghost of line {}'.format(i))

        n_lines_for_bestword = len(ratios[bestword]['line_list'])
        if n_lines_for_bestword == 0:
            del ratios[bestword]
            logger.debug(
                'No lines left for word {}, deleting it. {} words left'.format(
                    bestword, len(ratios)))
            wordindex = [i for i, tup in enumerate(wge) if tup[1] == bestword]
            del wge[wordindex[0]:wordindex[0] + 1]
            continue

        line_scores = [tup[0] for tup in ratios[bestword]['line_list']]
        insert = bisect.bisect(line_scores, score_threshold)
        if insert > len(line_scores):
            insert = len(line_scores)
        if insert == 0:
            maxupdate = 1
        else:
            maxupdate = insert
        if(batch):
            sqrt_lines = math.ceil(math.sqrt(len(line_scores)))
            if 2 * sqrt_lines > insert \
               and 2 * sqrt_lines <= len(line_scores):
                maxupdate = 2 * sqrt_lines
        logger.debug('insert = {}, max_update = {}  n_lines = {}'.format(
            insert, maxupdate, len(line_scores)))

        for pos in range(maxupdate):
            i = maxupdate - pos - 1
            lineid = ratios[bestword]['line_list'][i][2]
            if lineid not in pool:
                del ratios[bestword]['line_list'][i]
                logger.debug('Pruned the ghost of line {}'.format(lineid))
                continue
            sge = 0
            tokens = pool[lineid]['string'].split()
            count = count_tokens(tokens)
            for token in count.keys():
                sge += (ratios[token]['hconstant'] *
                        math.log((currmodel[token]['count'] + smoothing) /
                                 (currmodel[token]['count'] + count[token])))
            pool[first_bestword_id]['SGE'] = sge
            score = penalty[len(tokens)] + sge
            ratios[bestword]['line_list'][i] = (score, sge, lineid)

        n_lines_for_bestword = len(ratios[bestword]['line_list'])
        if n_lines_for_bestword == 0:
            del ratios[bestword]
            logger.debug(
                'No lines left for word {}, deleting it. {} words left'.format(
                    bestword, len(ratios)))
            wordindex = [i for i, tup in enumerate(wge) if tup[1] == bestword]
            del wge[wordindex[0]:wordindex[0] + 1]
            continue

        ratios[bestword]['line_list'] = sorted(ratios[bestword]['line_list'],
                                               key=lambda x: x[0])

        goodlines = []
        if batch:
            new_sqrt_lines = math.ceil(
                math.sqrt(len(ratios[bestword]['line_list'])))
            goodlines = ratios[bestword]['line_list'][:new_sqrt_lines]
            ratios[bestword]['line_list'] = \
                ratios[bestword]['line_list'][new_sqrt_lines:]

            to_prune_again = []
            for tup in goodlines:
                lineid = tup[2]
                if lineid not in pool:
                    to_prune_again.append(lineid)
            while(to_prune_again):
                i = to_prune_again.pop()
                del goodlines[i:i + 1]
                logger.debug('Pruned the ghost of line {}'.format(i))
            logger.debug('Adding {} of {} lines to selected data'.format(
                len(goodlines), n_lines_for_bestword))
        else:
            goodlines.append(ratios[bestword]['line_list'].pop(0))
            logger.debug('Adding {} of {} lines to selected data'.format(
                len(goodlines), n_lines_for_bestword))

        while goodlines:
            currmodel_linecount += 1
            tup = goodlines.pop(0)
            score, sge, lineid = tup
            if lineid not in pool:
                logger.debug('Ignored the ghost of line {}'.format(lineid))
                continue
            tokens = pool[lineid]['string'].split()
            currmodel_score += score
            currmodel_wordcount += len(tokens)
            selected.append(
                '{} {} {} {} {} {} {} {} {}'.format(
                    currmodel_score, score, penalty[len(tokens)], sge,
                    currmodel_linecount, lineid + 1, bestword, WGE,
                    pool[lineid]['string']))
            del pool[lineid]
            count = count_tokens(tokens)
            for token in count.keys():
                currmodel[token]['count'] += count[token]

        n_lines_for_bestword = len(ratios[bestword]['line_list'])
        if n_lines_for_bestword == 0:
            del ratios[bestword]
            logger.debug(
                'No lines left for word {}, deleting it. {} words left'.format(
                    bestword, len(ratios)))
            wordindex = [i for i, tup in enumerate(wge) if tup[1] == bestword]
            del wge[wordindex[0]:wordindex[0] + 1]
            continue

        wge = []
        re_band = re.compile(r'^@@BORING__0{1,7}')
        for word in ratios.keys():
            ratios[word]['WGE'] = ratios[word]['hconstant'] * \
                    math.log((currmodel[word]['count'] + smoothing) /
                             (currmodel[word]['count'] + 1))
            currmodel[word]['prob'] = currmodel[word]['count'] / \
                currmodel_wordcount
            if word == '@@DUBIOUS' \
               or word == '@@USELESS' \
               or word == '@@IMPOSSIBLE' \
               or word.startswith('@@-') \
               or re_band.match(word) is not None:
                continue
            wge.append((ratios[word]['WGE'], word, ratios[word]['hconstant']))

        if len(wge) == 0:
            logger.debug('Out of words!')
            break
        wge = sorted(wge, key=lambda x: x[0])

        for i in range(len(penalty)):
            penalty[i] = math.log((currmodel_wordcount + i) /
                                  currmodel_wordcount)

        maxiter -= 1
        linecount += 1

    return selected


def main():
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logger.setLevel('DEBUG')

    logfile = logging.FileHandler(
                    time.strftime('{}-{}-%Y%m%d_%H%M.log'.format(
                        args.task, args.unadapted)),
                    encoding='utf-8')
    logfile.setLevel('DEBUG')
    formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(message)s')
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)

    with open(args.task, 'r') as task_file:
        logger.debug('Loading task data {}'.format(args.task))
        task_data = []
        for line in task_file:
            if args.lower:
                task_data.append(line.strip().lower())
            else:
                task_data.append(line.strip())

    with open(args.unadapted, 'r') as unadapted_file:
        logger.debug('Loading unadapted data {}'.format(args.unadapted))
        unadapted_data = []
        for line in unadapted_file:
            if args.lower:
                unadapted_data.append(line.strip().lower())
            else:
                unadapted_data.append(line.strip())

    logger.debug('Computing vocabulary counts')
    task_vocab = compute_counts(task_data)
    unadapted_vocab = compute_counts(unadapted_data)
    del task_data

    logger.debug('Computing ratios between both vocabularies')
    ratios, sizes = compute_ratios(task_vocab, unadapted_vocab)
    del task_vocab, unadapted_vocab

    logger.debug('Squishing ratios over uninteresting vocab')
    currmodel = {}
    ratios, replace = squish_ratios(ratios, args.mincount, args.keep)
    ratios, currmodel = init_model(ratios, sizes, currmodel)

    logger.debug('Squishing unadapted data')
    unadapted_squish = squish_corpus(unadapted_data, replace)
    del unadapted_data, replace

    logger.debug('Index unadapted sentences and init the model')
    penalty = init_penalty(args.maxlen, args.smoothing)
    pool, ratios = index_unadapted(unadapted_squish, ratios,
                                   args.smoothing, penalty, logger)
    wge = init_estimates(ratios, args.smoothing)
    del unadapted_squish

    logger.debug('Perform selection')
    selected = []
    selected = main_loop(selected, pool, ratios, currmodel, penalty, wge,
                         args.smoothing, args.batch, logger)

    for line in selected:
        print(line)


if __name__ == '__main__':
    main()
