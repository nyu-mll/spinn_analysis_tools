"""

Script to generate a spreadsheet of results for large experiment sweeps.

Example command,
python sweep_results.py --log_path /scratch/nn1119/nli/ --log_type spinn --sort_column best_dev --path_out ../results

"""


import glob
import os
import gflags
import sys
import re

import pandas as pd

FLAGS = gflags.FLAGS
gflags.DEFINE_string("log_path", "../python/logs", "Folder with log files.")
gflags.DEFINE_string("log_type", "", "Filter log files with a string presen in file name.")
gflags.DEFINE_string("sort_column", "model_type", "Sort dataframe by this column.")
gflags.DEFINE_string("path_out", "", "Path to optional excel file.")
FLAGS(sys.argv)

if FLAGS.log_type:
	ltype = "*" + FLAGS.log_type + "*.log"
else:
	ltype = "*.log"
path = os.path.join(FLAGS.log_path, ltype)
files = glob.glob(path)

params = ["model_type", "model_dim", "learning_rate", "l2_lambda", "learning_rate_decay_when_no_progress", "optimizer_type"]

results = {}
for file in files:
	### Populate hyperparameters ###
	with open(file) as f:
		p_vals = dict((p, []) for p in params)
		proceed = sum([int(len(p_vals[p]) > 0) for p in params])
		for line in f:
			for p in params:
				p_ = p + "_"
				_p = "_" + p
				s = re.search(p, line)
				s_ = re.search(p_, line)
				_s = re.search(_p, line)
				if s and not s_ and not _s:
					pal = []
					val = p + "\": (.*)"
					pal = re.search(val, line)
					if pal:
						pal = pal.groups()[0].replace(",", "")
					p_vals[p].append(pal)

			proceed = sum([int(len(p_vals[p]) > 0) for p in params])
			if proceed == len(params):
				break

	### Find best checkpoint ###
	best_string = "Checkpointing with new best dev accuracy of (.*)"
	for line in reversed(list(open(file))):
		best = re.search(best_string, line)
		if best:
			best_dev = best.groups()[0]
			p_vals["best_dev"] = best_dev
		if "best_dev" in p_vals.keys():
			break
	p_vals["file"] = file
	results[file] = p_vals


### Convert to pandas dataframe ### 
results_val = [v for v in results.values()]
df = pd.DataFrame(results_val)
df = df.rename(columns={"learning_rate_decay_when_no_progress" : "lr_decay"})


df = df.sort_values(FLAGS.sort_column)
with pd.option_context('display.max_rows', None, 'display.max_columns', 10):
    print(df)

if FLAGS.path_out:
	df.to_excel(FLAGS.path_out)
