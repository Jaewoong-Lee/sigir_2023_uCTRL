import numpy as np

# bpr_tuning_result_cm-100k
# bpr_tuning_result_kuai

# direct_tuning_result_cm-100k
# direct_tuning_result_kuai

# macr_tuning_result_cm-100k
# macr_tuning_result_kuai

# ssm_tuning_result_cm-100k
# ssm_tuning_result_kuai


name = 'ssm_tuning_result_kuai'
f = open("./%s"%(name), 'r')
lines = f.readlines()

hyper = []
valid = []
test = []
for i, line in enumerate(lines):
    line = line.rstrip()
    # line = line.split(",")

    if i % 6 == 0:
        hyper.append(line)
    if i % 6 == 2:
        valid.append(line)
    if i % 6 == 4:
        test.append(line)
f.close()


valid = np.asarray(valid)
test = np.asarray(test)
hyper = np.asarray(hyper)


for_sorting = []
for sss in valid:
    line = sss.split(" ")
    print(line[26])
    for_sorting.append(float(line[26]))

sorted_index = np.argsort(for_sorting)
sorted_index = sorted_index[::-1]

valid = valid[sorted_index]
test = test[sorted_index]
hyper = hyper[sorted_index]



with open("./%s_re"%(name), 'w') as f:
    for i, line in enumerate(valid):
        line_str = line + "\t"
        f.write(line_str)

        line_str = test[i] + "\t"
        f.write(line_str)

        line_str = hyper[i] + "\n"
        f.write(line_str)
