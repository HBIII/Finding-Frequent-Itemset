from collections import defaultdict
from pyspark import SparkContext
from operator import add
import time
import sys

start_time = time.time()

sc = SparkContext()

support = int(sys.argv[2])

text = sc.textFile(sys.argv[3])
header = text.first()
text = text.filter(lambda line: line != header)
text = text.map(lambda line: line.split(","))
candidate_string = "Candidates:\n"

if sys.argv[1] == "1":
    user_businesses = text.map(lambda user_business: (user_business[0], {user_business[1]})).reduceByKey(lambda x, y: x | y)
else:
    user_businesses = text.map(lambda user_business: (user_business[1], {user_business[0]})).reduceByKey(lambda x, y: x | y)

businesses = user_businesses.map(lambda user_business: user_business[1])
businesses_len = businesses.count()

def single_candidate(business, support_threshold):

    single_candidate_set = set()
    count_dict = {}
    single_candidate = set()

    for items in business:
        for item in items:
            if item in count_dict:
                count_dict[item] += 1
            else:
                count_dict[item] = 1

    for i in count_dict:
        if count_dict[i] >= support_threshold:
            single_candidate.add(i)

    single_candidate = sorted(single_candidate)

    for elem in single_candidate:
        single_candidate_set.add((elem,))

    pair_combinations = set()

    for i in range(len(single_candidate)):
        for j in range(i + 1, len(single_candidate)):
            pair_combinations.add((single_candidate[i], single_candidate[j]))

    pair_combinations = set(sorted(pair_combinations))
    return pair_combinations, single_candidate_set


def pair_candidate(business,pair_combinations,support_threshold):
    candidate_pairs = []

    for items in pair_combinations:
        counter = 0
        for line in business:
            candidate = set(items)
            for item in items:
                if item in line:
                    candidate.remove(item)
                if not candidate:
                    counter += 1
                    break
        if counter >= support_threshold:
            candidate_pairs.append(tuple(items))

    candidate_pairs_sorted = set(sorted(candidate_pairs))
    return candidate_pairs_sorted

def combination_candidate(business, candidate_pairs_sorted, support_threshold):
    candidates_combination = set()

    while candidate_pairs_sorted:

        prev_combinations = list(candidate_pairs_sorted)
        combination_set = set()

        for i in range(len(candidate_pairs_sorted)):
            for j in range(i + 1, len(candidate_pairs_sorted)):
                new_candidate = set()

                new_candidate.update(prev_combinations[i])
                new_candidate.update(prev_combinations[j])

                if new_candidate in combination_set:
                    continue
                if len(new_candidate) == len(prev_combinations[i]) + 1:
                    combination_set.add(tuple(sorted(new_candidate)))

        combination_set = list(combination_set)
        combination_set.sort()

        if len(combination_set) == 0:
            break

        candidate_pairs = []

        for items in combination_set:
            counter = 0
            for line in business:
                candidate = set(items)
                for item in items:
                    if item in line:
                        candidate.remove(item)
                    if not candidate:
                        counter += 1
                        break
            if counter >= support_threshold:
                candidate_pairs.append(tuple(items))

        candidate_pairs_sorted = set(sorted(candidate_pairs))
        candidates_combination.update(candidate_pairs_sorted)

        if not candidate_pairs_sorted:
            break

    return candidates_combination

def get_candidate_counts(user_business, candidate_itemsets):
    item_counts = defaultdict(int)
    for user, business in user_business:
        for item in candidate_itemsets:
            if set(business).issuperset(item):
                item_counts[item] += 1
    for item in sorted(item_counts.keys()):
        yield (item, item_counts[item])

def apriori_algorithm(business):
    business = [line for line in business]
    n = len(business)

    partition_support = (float(n)/businesses_len) * support
    
    candidates_set = set()

    pair_combinations, single_candidate_set = single_candidate(business, partition_support)
    candidates_set.update(single_candidate_set)

    candidate_pairs_sorted = pair_candidate(business, pair_combinations, partition_support)
    candidates_set.update(candidate_pairs_sorted)

    candidate_set = combination_candidate(business, candidate_pairs_sorted, partition_support)
    candidates_set.update(candidate_set)
    
    return sorted(candidates_set)

candidate_superset = businesses.mapPartitions(lambda business: apriori_algorithm(business)).distinct()

candidate_itemsets = candidate_superset.groupBy(lambda x: len(x)).collectAsMap()

for key in sorted(candidate_itemsets.keys()):
    itemsets = candidate_itemsets[key]
    sorted_itemsets = sorted([sorted(itemset) for itemset in itemsets])
    line = ",".join([str(tuple(items)) for items in sorted_itemsets]).replace(r',)', ')')
    candidate_string += line +'\n\n'

outputFile = open(sys.argv[4], "w")
outputFile.write(candidate_string)

frequent_string = "Frequent Itemsets:\n"

candidate_superset = candidate_superset.collect()
# print(list(candidate_superset))

frequent_itemsets = user_businesses.mapPartitions(lambda user_business:
                                                  get_candidate_counts(user_business, candidate_superset))\
    .reduceByKey(add).filter(lambda candidate: candidate[1] >= support).map(lambda x: x[0]).groupBy(len).collectAsMap()

for key in sorted(frequent_itemsets.keys()):
    itemsets = frequent_itemsets[key]
    sorted_itemsets = sorted([sorted(itemset) for itemset in itemsets])
    line = ",".join([str(tuple(items)) for items in sorted_itemsets]).replace(r',)', ')')
    frequent_string += line +'\n\n'

outputFile.write(frequent_string.rstrip())

print("Duration: ", time.time() - start_time)
