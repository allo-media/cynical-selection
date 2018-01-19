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
                    help='Unadapted sentences, the pool we want to pick from')
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
parser.add_argument('--iterate', action='store_true', dest='iterate',
                    help=("Set this flag to iterate until we can't reduce " +
                          "selected data by no more than 10%"))
parser.add_argument('--no-iterate', action='store_false', dest='iterate',
                    help='Set this flag to iterate on data only once')
parser.set_defaults(iterate=False)


def get_logger(args):
    """Returns a file logger with DEBUG level.

    args: argparse object containing the cmd arguments

    returns: logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel('DEBUG')

    logfile = logging.FileHandler(
                    time.strftime('{}-{}-%Y%m%d_%H%M.log'.format(
                        args.unadapted, args.task)), encoding='utf-8')
    logfile.setLevel('DEBUG')
    formatter = logging.Formatter(
                    '%(asctime)s | %(message)s')
    logfile.setFormatter(formatter)
    logger.addHandler(logfile)

    return logger


def compute_counts(corpus):
    """Compute the word counts and probs for a given corpus

    corpus: list of sentences

    returns: dict of words, containing counts & probs
    """
    words = {}
    size = 0

    # Let's count words first
    for line in corpus:
        for token in line.split():
            if token in words:
                words[token]['count'] += 1
            else:
                words[token] = {}
                words[token]['count'] = 1
            size += 1

    # Then we compute all the probs once we know the final size
    for k in words.keys():
        words[k]['prob'] = words[k]['count'] / size

    return words


def float_to_str(num):
    """Gets rid of exponent notation in floats

    num: the float to return as a string
    """
    with localcontext() as ctx:
        ctx.prec = 20
        d = ctx.create_decimal(repr(num))
    return format(d, 'f')


def compute_ratios(task_vocab, unadapted_vocab):
    """Computes the stats between the two vocabularies

    task_vocab: dict of task words with count and prob
    unadapted_vocab: dict of unadapted words with count and prob

    returns:
        ratios: initialized dict (word-indexed) with computed stats
        sizes: dict of task & unadapted data sizes
    """
    ratios = {}
    sizes = {}
    sizes['task'] = 0
    sizes['unadapted'] = 0

    # We iterate on the unadapted vocab first
    for word in unadapted_vocab.keys():
        ratios[word] = {}
        # Word is in task vocab, let's compute the delta
        # and add the probs and counts
        if word in task_vocab:
            ratios[word]['delta'] = (task_vocab[word]['prob'] /
                                     unadapted_vocab[word]['prob'])
            ratios[word]['t_prob'] = task_vocab[word]['prob']
            ratios[word]['t_count'] = task_vocab[word]['count']
            sizes['task'] += task_vocab[word]['count']
            del task_vocab[word]
        # Minimal delta, prob & count are 0
        else:
            ratios[word]['delta'] = 0.5 / unadapted_vocab[word]['count']
            ratios[word]['t_prob'] = 0
            ratios[word]['t_count'] = 0
        # Add the stats for the unadapted part
        ratios[word]['u_prob'] = unadapted_vocab[word]['prob']
        ratios[word]['u_count'] = unadapted_vocab[word]['count']
        sizes['unadapted'] += unadapted_vocab[word]['count']

    # Let's add the "orphan" task words with delta = count * 2
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
    """Reduces the ratios to interesting words with some clustering.

    ratios: the dict of stats
    mincount: min count of given word in each vocab to be taken into account
    keep: do we keep the boring words as is or do we cluster them?

    returns: updated ratios dict & words replacement dict
    """
    replace = {}
    # Group every 0 after the decimal dot (used in boring clustering)
    re_band = re.compile(r'.*\.(?P<band>0*).*')
    # Impossible words are task words not in unadapted
    ratios['@@IMPOSSIBLE'] = {}
    # Useless words are unadapted words not in task
    ratios['@@USELESS'] = {}
    # Dubious words are under the mincount
    ratios['@@DUBIOUS'] = {}
    # t_count is task words, u_count is unadapted words
    ratios['@@IMPOSSIBLE']['t_count'] = 0
    ratios['@@USELESS']['t_count'] = 0
    ratios['@@DUBIOUS']['t_count'] = 0
    ratios['@@IMPOSSIBLE']['u_count'] = 0
    ratios['@@USELESS']['u_count'] = 0
    ratios['@@DUBIOUS']['u_count'] = 0

    for word in list(ratios.keys()):
        # Don't mess with the special words while we create them
        if word.startswith('@@'):
            continue
        # Squished words are deleted, but they are kept in the replace dict
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
        # Words between a certain delta are either bucketed
        # into boring words or kept as is
        elif ratios[word]['delta'] < math.exp(1) \
                and ratios[word]['delta'] > math.exp(-1) \
                and not keep:
            # Get the band name (number of 0's)
            band = re_band.sub(r'\g<band>',
                               float_to_str(ratios[word]['u_prob']))
            bucket = '@@BORING__' + band
            replace[word] = bucket
            # Create the bucket if it doesn't exists
            if bucket not in ratios:
                ratios[bucket] = {}
                ratios[bucket]['t_count'] = 0
                ratios[bucket]['u_count'] = 0
            ratios[bucket]['delta'] = bucket
            ratios[bucket]['t_count'] += ratios[word]['t_count']
            ratios[bucket]['u_count'] += ratios[word]['u_count']
            del ratios[word]
        # Log-negative words (delta < 1)
        # They are bucketed into the form @@-1, @@-2, etc.
        # (Truncation on the integer part)
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
        # Else we keep it as is
        else:
            replace[word] = word

    return ratios, replace


def init_model(ratios, sizes):
    """Initialize the hconstant parameter for each word
       and populate the model dict

    ratios: the ratios dict, nothing new
    sizes: the sizes dict

    returns: the updated ratios dict & the initialized model
    """
    model = {}
    for word in ratios.keys():
        ratios[word]['hconstant'] = ratios[word]['t_count'] / sizes['task']
        if word not in model:
            model[word] = {}
            model[word]['prob'] = 0
            model[word]['count'] = 0
    return ratios, model


def squish_corpus(corpus, replace):
    """Squishes a corpus of data using a replacement dict

    corpus: text data to squish
    replace: the replacement dict

    returns: the squished corpus (list of strings)
    """
    squished = []
    for line in corpus:
        squished.append(' '.join([replace[token] for token in line.split()]))
    return squished


def init_penalty(maxlen, smoothing):
    """Initializes the penalty used in cross-entropy computation

    maxlen: maximum allowed length of a sentence
    smoothing: smoothing factor to avoid dividing by 0 (default 0.01)

    returns: the penalty list
    """
    penalty = []
    for i in range(maxlen):
        penalty.append(math.log((i + 2 * smoothing) / smoothing))
    return penalty


def count_tokens(tokens):
    """Count words into token lists

    tokens: list of tokens (typîcally a sentence) to count

    returns: dict of words with individual counts
    """
    count = {}
    for token in tokens:
        if token in count:
            count[token] += 1
        else:
            count[token] = 1
    return count


def index_unadapted(squish, ratios, smoothing, penalty, logger):
    """Index the squished unadapted sentences in the ratios dict

    squish: the squished unadapted sentences list
    ratios: the ratios dict
    smoothing: the smoothing factor
    penalty: the initialized penalty list
    logger: logging facility

    returns:
        pool: the pool of available sentences
        ratios: the updated ratios dict
    """
    pool = {}

    for lineid, line in enumerate(squish):
        tokens = line.split()
        # Ignore longer sentences
        if len(tokens) > len(penalty):
            continue
        # Put the sentence and its length in the pool
        pool[lineid] = {}
        pool[lineid]['string'] = line
        pool[lineid]['count'] = len(tokens)

        sge = 0
        count = count_tokens(tokens)
        # Compute the initial sentence gain estimate (SGE)
        for token in count.keys():
            sge += (ratios[token]['hconstant'] * math.log(smoothing /
                                                          count[token]))
            # Score = penalty (positive) + gain (negative)
            # We aim at sentences whose score < 0
            score = penalty[len(tokens) - 1] + sge
            if 'line_list' not in ratios[token]:
                ratios[token]['line_list'] = []
            # Append sentence score, SGE and pool lineid
            # to the word's list of sentences
            ratios[token]['line_list'].append((score, sge, lineid))
        pool[lineid]['SGE'] = sge

    # Sort the lists of sentences by score
    for word in list(ratios.keys()):
        if 'line_list' in ratios[word]:
            ratios[word]['line_list'] = sorted(ratios[word]['line_list'],
                                               key=lambda x: x[0])
        else:
            # Delete the words without lines
            del ratios[word]
            logger.debug('deleted {}, no lines'.format(word))

    return pool, ratios


def init_wge(ratios, smoothing):
    """Initializes the word gain estimates (WGE)

    ratios: the ratios dict
    smoothing: the smoothing factor

    returns: the word gain estimates list
    """
    wge = []
    # Compile the @@BORING matching regex
    re_band = re.compile(r'^@@BORING__0{1,7}')

    for token in ratios.keys():
        # Init the WGE
        ratios[token]['WGE'] = ratios[token]['hconstant'] * \
                math.log(smoothing / 1)
        # Ignore unwanted words
        if token == '@@DUBIOUS' \
           or token == '@@USELESS' \
           or token == '@@IMPOSSIBLE' \
           or token.startswith('@@-') \
           or re_band.match(token) is not None:
            continue
        # Add the WGE to teh list
        wge.append((ratios[token]['WGE'], token, ratios[token]['hconstant']))
    # And sort it, of course
    wge = sorted(wge, key=lambda x: x[0])

    return wge


def select_data(pool, ratios, model, penalty,
                wge, smoothing, batch, logger):
    """The real deal, this is where we select, baby!

    pool: the unadapted sentences pool dict
    ratios: the ratios dict
    model: our counts model
    penalty: the penalty list
    wge: the word gain estimates list
    smoothing: the smoothing factor
    batch: the batching flag
    logger: logging facility

    returns: the selected sentences list
    """
    selected = []
    maxiter = len(pool)
    linecount = 1
    model_linecount = 0
    model_wordcount = 0
    model_score = 0
    # Can't select more sentences than we have!
    logger.debug('Running for maximum {} iterations'.format(maxiter))
    while(maxiter > 0):
        # No WGE means no interesting words anymore
        if not wge:
            logger.debug('wge is empty: {}'.format(wge))
            logger.debug('words left: {}'.format(ratios.keys()))
            logger.debug("I think we're done here.")
            break
        # Get the best WGE tuple
        (WGE, bestword, hconstant) = wge[0]
        logger.debug('best word: {} (WGE={})'.format(bestword, WGE))
        n_lines_for_bestword = len(ratios[bestword]['line_list'])
        # If we don't have lines for this word, no need to continue with it
        if n_lines_for_bestword == 0:
            del ratios[bestword]
            logger.debug(
                'No lines left for word {}, deleting it. {} words left'.format(
                    bestword, len(ratios)))
            wordindex = [i for i, tup in enumerate(wge) if tup[1] == bestword]
            del wge[wordindex[0]:wordindex[0] + 1]
            continue

        # Find the line id of the best sentence of our best word
        first_bestword_id = 0
        indices_to_prune = []
        score_threshold = 0
        for i in range(n_lines_for_bestword):
            # Get the best id
            first_bestword_id = ratios[bestword]['line_list'][i][2]
            # The sentence still exists in the pool, perfect
            if first_bestword_id in pool:
                tokens = pool[first_bestword_id]['string'].split()
                sge = 0
                count = count_tokens(tokens)
                # Update this sentence SGE & set the score as our threshold
                for token in count.keys():
                    sge += (ratios[token]['hconstant'] *
                            math.log((model[token]['count'] + smoothing) /
                                     (model[token]['count'] +
                                      count[token])))
                pool[first_bestword_id]['SGE'] = sge
                score_threshold = penalty[len(tokens) - 1] + sge
                logger.debug('SGE: {} => {}'.format(
                    ratios[bestword]['line_list'][i][1], sge))
                logger.debug('score: {} => {}'.format(
                    ratios[bestword]['line_list'][i][0], score_threshold))
                # We have our sentence, don't have to look anymore
                break
            # We don't have the sentence in the pool, let's mark it for pruning
            else:
                indices_to_prune.append(i)

        # Remove all the marked sentences
        while indices_to_prune:
            i = indices_to_prune.pop()
            del ratios[bestword]['line_list'][i]
            logger.debug('Pruned the ghost of line {}'.format(i))

        # Sanity check, do we have any lines left?
        n_lines_for_bestword = len(ratios[bestword]['line_list'])
        if n_lines_for_bestword == 0:
            del ratios[bestword]
            logger.debug(
                'No lines left for word {}, deleting it. {} words left'.format(
                    bestword, len(ratios)))
            wordindex = [i for i, tup in enumerate(wge) if tup[1] == bestword]
            del wge[wordindex[0]:wordindex[0] + 1]
            continue

        # Determine where our sentence would be inserted with its new score
        line_scores = [tup[0] for tup in ratios[bestword]['line_list']]
        insert = bisect.bisect(line_scores, score_threshold)
        # Some checks on the insertion position
        if insert > len(line_scores):
            insert = len(line_scores)
        if insert == 0:
            maxupdate = 1
        else:
            maxupdate = insert
        # Batch mode: we update 2 * sqrt(total lines)
        # if > insert, of course, and if < num lines, of course too
        if(batch):
            sqrt_lines = math.ceil(math.sqrt(len(line_scores)))
            if 2 * sqrt_lines > insert \
               and 2 * sqrt_lines <= len(line_scores):
                maxupdate = 2 * sqrt_lines
        logger.debug('insert = {}, max_update = {}  n_lines = {}'.format(
            insert, maxupdate, len(line_scores)))

        # Update all sentences till max update
        for pos in range(maxupdate):
            # We go backwards here, to not mess the indices
            i = maxupdate - pos - 1
            lineid = ratios[bestword]['line_list'][i][2]
            # Ghost pruning here, again
            if lineid not in pool:
                del ratios[bestword]['line_list'][i]
                logger.debug('Pruned the ghost of line {}'.format(lineid))
                continue
            # update SGE & score in the ratios dict
            sge = 0
            tokens = pool[lineid]['string'].split()
            count = count_tokens(tokens)
            for token in count.keys():
                sge += (ratios[token]['hconstant'] *
                        math.log((model[token]['count'] + smoothing) /
                                 (model[token]['count'] + count[token])))
            pool[first_bestword_id]['SGE'] = sge
            score = penalty[len(tokens) - 1] + sge
            ratios[bestword]['line_list'][i] = (score, sge, lineid)

        # Sanity check again, do we have any lines left?
        n_lines_for_bestword = len(ratios[bestword]['line_list'])
        if n_lines_for_bestword == 0:
            del ratios[bestword]
            logger.debug(
                'No lines left for word {}, deleting it. {} words left'.format(
                    bestword, len(ratios)))
            wordindex = [i for i, tup in enumerate(wge) if tup[1] == bestword]
            del wge[wordindex[0]:wordindex[0] + 1]
            continue

        # Sort our updated lines list
        ratios[bestword]['line_list'] = sorted(ratios[bestword]['line_list'],
                                               key=lambda x: x[0])

        goodlines = []
        # Batch mode: we update sqrt(total lines) lines
        if batch:
            new_sqrt_lines = math.ceil(
                math.sqrt(len(ratios[bestword]['line_list'])))
            # Manual splicing (lol)
            goodlines = ratios[bestword]['line_list'][:new_sqrt_lines]
            ratios[bestword]['line_list'] = \
                ratios[bestword]['line_list'][new_sqrt_lines:]

            # Maybe we have some ghosts to prune (again),
            # so let's do it now
            to_prune_again = []
            for i, tup in enumerate(goodlines):
                lineid = tup[2]
                if lineid not in pool:
                    to_prune_again.append(i)
            while(to_prune_again):
                i = to_prune_again.pop()
                del goodlines[i]
                logger.debug('Pruned the ghost of line {}'.format(i))
            logger.debug('Adding {} of {} lines to selected data'.format(
                len(goodlines), n_lines_for_bestword))
        # No batch mode, take the first line
        else:
            goodlines.append(ratios[bestword]['line_list'].pop(0))
            logger.debug('Adding {} of {} lines to selected data'.format(
                len(goodlines), n_lines_for_bestword))

        # Finally, add our goodlines to the selected list
        while goodlines:
            model_linecount += 1
            tup = goodlines.pop(0)
            score, sge, lineid = tup
            # Are we sure this isn't a ghost?
            if lineid not in pool:
                logger.debug('Ignored the ghost of line {}'.format(lineid))
                continue
            tokens = pool[lineid]['string'].split()
            # Update the model
            model_score += score
            model_wordcount += len(tokens)
            # Add our sentence(s) to the selected list
            # and delete it from the pool
            # Format is:
            #   model score, sentence score, length penalty,
            #   sentence gain estimate, line id in the model,
            #   line id in the original data, best word,
            #   best word WGE, the squished sentence
            selected.append(
                '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
                    model_score, score, penalty[len(tokens) - 1], sge,
                    model_linecount, lineid + 1, bestword, WGE,
                    pool[lineid]['string']))
            del pool[lineid]
            # Update the model count
            count = count_tokens(tokens)
            for token in count.keys():
                model[token]['count'] += count[token]

        # One last sanity check before we go!
        n_lines_for_bestword = len(ratios[bestword]['line_list'])
        if n_lines_for_bestword == 0:
            del ratios[bestword]
            logger.debug(
                'No lines left for word {}, deleting it. {} words left'.format(
                    bestword, len(ratios)))
            wordindex = [i for i, tup in enumerate(wge) if tup[1] == bestword]
            del wge[wordindex[0]:wordindex[0] + 1]
            continue

        # Update the WGE list for the next round
        wge = []
        re_band = re.compile(r'^@@BORING__0{1,7}')
        for word in ratios.keys():
            ratios[word]['WGE'] = ratios[word]['hconstant'] * \
                    math.log((model[word]['count'] + smoothing) /
                             (model[word]['count'] + 1))
            model[word]['prob'] = model[word]['count'] / \
                model_wordcount
            if word == '@@DUBIOUS' \
               or word == '@@USELESS' \
               or word == '@@IMPOSSIBLE' \
               or word.startswith('@@-') \
               or re_band.match(word) is not None:
                continue
            wge.append((ratios[word]['WGE'], word, ratios[word]['hconstant']))

        # No point in continuing...
        if len(wge) == 0:
            logger.debug('Out of words!')
            break

        # Sort our updated WGE and update our penalties
        wge = sorted(wge, key=lambda x: x[0])
        for i in range(len(penalty)):
            penalty[i] = math.log((model_wordcount + i) /
                                  model_wordcount)

        maxiter -= 1
        linecount += 1

    return selected


def main_loop(task_data, unadapted_data, args, logger):
    """Our main selection loop, which can be executed a number of times

    task_data: our list of task sentences
    unadapted_data: our list of unadapted sentences
    args: argparse object
    logger: logging facility

    returns: the selected sentences
    """
    logger.debug('Computing vocabulary counts')
    task_vocab = compute_counts(task_data)
    unadapted_vocab = compute_counts(unadapted_data)
    del task_data

    logger.debug('Computing ratios between both vocabularies')
    ratios, sizes = compute_ratios(task_vocab, unadapted_vocab)
    del task_vocab, unadapted_vocab

    logger.debug('Squishing ratios over uninteresting vocab')
    ratios, replace = squish_ratios(ratios, args.mincount, args.keep)
    ratios, model = init_model(ratios, sizes)

    logger.debug('Squishing unadapted data')
    unadapted_squish = squish_corpus(unadapted_data, replace)
    del unadapted_data, replace

    logger.debug('Index unadapted sentences and init the model')
    penalty = init_penalty(args.maxlen, args.smoothing)
    pool, ratios = index_unadapted(unadapted_squish, ratios,
                                   args.smoothing, penalty, logger)
    wge = init_wge(ratios, args.smoothing)
    del unadapted_squish

    logger.debug('Perform selection')
    selected = select_data(pool, ratios, model, penalty, wge,
                           args.smoothing, args.batch, logger)

    return selected


def load_data(data_file, lower, logger):
    """Loads data from a text file into a list

    data_file: the text file to load
    lower: flag for lowercasing
    logger: logging facility

    returns: the loaded data in a list
    """
    with open(data_file, 'r') as handle:
        logger.debug('Loading data file {}'.format(data_file))
        data = []
        for line in handle:
            if lower:
                data.append(line.strip().lower())
            else:
                data.append(line.strip())
    return data


def unsquish(selected, unadapted_data):
    """Transform our selected squished text back to the original text

    selected: the selected sentences
    unadapted_data: the list of original unadapted sentences

    returns: the selected sentences, with clear text
    """
    for i, line in enumerate(selected):
        items = line.split('\t')
        items[8] = unadapted_data[int(items[5]) - 1]
        selected[i] = '\t'.join(items)

    return selected


def extract_data(selected):
    """Returns the text data from a selected sentences list
    """
    data = []
    for line in selected:
        data.append(line.split('\t')[8])

    return data


def main():
    args = parser.parse_args()
    logger = get_logger(args)
    outname = '{}-{}.jaded'.format(args.unadapted, args.task)

    # Load our data
    task_data = load_data(args.task, args.lower, logger)
    unadapted_data = load_data(args.unadapted, args.lower, logger)

    # Stop criterion
    # we stop when we're unable to remove more than 10% of data
    max_selection = int(len(unadapted_data) * 0.9)
    selected = main_loop(task_data, unadapted_data, args, logger)

    if args.iterate:
        # While we remove more than 10%, let's continue!
        while (len(selected) < max_selection):
            unadapted_data = extract_data(unsquish(selected, unadapted_data))
            max_selection = int(len(unadapted_data) * 0.9)
            selected = main_loop(task_data, unadapted_data, args, logger)

    # Write our output, the much awaited selected sentences!
    with open(outname, 'w') as out:
        for line in unsquish(selected, unadapted_data):
            out.write(line + '\n')


if __name__ == '__main__':
    main()
