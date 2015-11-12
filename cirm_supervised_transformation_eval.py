#! /usr/bin/env python2.7

# Example:
#./cirm_supervised_transformation_eval.py --in_data1 test1.tab --in_data2 test2.tab --in_clin1 clin1.tab --in_clin2 clin2.tab --attribute tissue --out_file_prefix path/
#python2.7 cirm_supervised_transformation_eval.py --in_data1 test1.tab --in_data2 test2.tab --in_clin1 clin1.tab --in_clin2 clin2.tab --attribute tissue --out_file_prefix path/
#
#python2.7 /Users/ynewton/school/ucsc/projects/stuart_lab/scripts/cirm_supervised_transformation_eval.py --in_data1 /Users/ynewton/school/ucsc/projects/stuart_lab/CIRM/untransformed/tissue_atlas.reduced.nodups.tab --in_data2 /Users/ynewton/school/ucsc/projects/stuart_lab/CIRM/untransformed/GTEx_Analysis_2014-01-17_RNA-seq_RNA-SeQCv1.1.8_gene_rpkm.reduced.nodups.tab --in_clin1 /Users/ynewton/school/ucsc/projects/stuart_lab/CIRM/tissue_atlas_annotation.reduced.tab --in_clin2 /Users/ynewton/school/ucsc/projects/stuart_lab/CIRM/gtex_annotation.reduced.tab --attribute tissue --out_file_prefix /Users/ynewton/school/ucsc/projects/stuart_lab/CIRM/untransformed/

import argparse
import sys
import os
import re
import math
import string
import copy
from datetime import datetime
import optparse

# SciKit-learn related includes
#import pylab as pl
#import matplotlib
import numpy as np
import sklearn
from sklearn import cross_validation, ensemble, feature_selection, lda, linear_model, metrics, neighbors , qda, svm, tree
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from scipy import linalg
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from matplotlib import rcParams

def read_tabular(input_file, numeric_flag):
	data = open(input_file, 'r') 	
	init_matrix = []
	col_headers = []
	row_headers = []
	line_num = 1
	for line in data:
		line_elems = line.strip().split("\t")
		if line_num == 1:
			col_headers = line_elems[1:]
		else:
			row_headers.append(line_elems[0])
			features = line_elems[1:]
			init_matrix.append(features)
			
		line_num += 1
	
	if numeric_flag:
		matrix = [map(float,x) for x in init_matrix]
	else:
		matrix = init_matrix
	return (matrix, col_headers, row_headers)

def read_input(input_file, input_classes, attribute):
	input = read_tabular(input_file, True)
	X =zip(*input[0])
	classes = read_tabular(input_classes, False)
	if not(attribute in classes[1]):
		print >> sys.stderr, "ERROR: specified attribute name is not in the input data"
	else:
		attribute_index = classes[1].index(attribute)
	Y = list(zip(*classes[0])[attribute_index])
	Y_set = set(Y)
	Y_list = list(Y_set)
	for val in Y_list:
		replace_val = Y_list.index(val)
		for k, v in enumerate(Y):
			if v == val:
				Y[k] = replace_val
	#select only those samples that are in the labels:
	common_samples = list(set(input[1]).intersection(classes[2]))
	X_new = []
	Y_new = []
	for cs in common_samples:
		input_position = input[1].index(cs)
		class_position = classes[2].index(cs)
		Y_new.append(Y[class_position])
		X_new.append(X[input_position])
	return(X_new, Y_new, Y_list, input[2])

def filter_on_class_counts(input_data, input_labels, min_count):
	new_input_data = []
	new_input_labels = []
	unique_labels = set(input_labels)
	for l in unique_labels:
		l_indices = [i for i, x in enumerate(input_labels) if x == l]
		if len(l_indices) >= min_count:
			for l_i in l_indices:
				new_input_data.append(input_data[l_i])
				new_input_labels.append(input_labels[l_i])
	return(new_input_data,new_input_labels)

def filter_on_common_genes(data1, data1_genes, data2, data2_genes):
	data1_t =zip(*data1)
	data2_t =zip(*data2)
	new_data1_t = []
	new_data2_t = []
	common_genes = list(set(data1_genes).intersection(data2_genes))
	for cg in common_genes:
		cg_index1 = data1_genes.index(cg)
		new_data1_t.append(data1_t[cg_index1])
		cg_index2 = data2_genes.index(cg)
		new_data2_t.append(data2_t[cg_index2])
	new_data1 = zip(*new_data1_t)
	new_data2 = zip(*new_data2_t)
	return(new_data1, new_data2, common_genes)

def filter_on_common_classes(data1, data1_classes, data1_class_labels, data2, data2_classes, data2_class_labels):
	new_data1_class_labels = []
	new_data2_class_labels = []
	for i in range(len(data1_class_labels)):
		if i in data1_classes:
			new_data1_class_labels.append(data1_class_labels[i])
	for i in range(len(data1_class_labels)):
		if i in data2_classes:
			new_data2_class_labels.append(data2_class_labels[i])
	common_labels = list(set(new_data1_class_labels).intersection(new_data2_class_labels))
	new_common_labels = []
	new_data1 = []
	new_data1_classes = []
	new_data2 = []
	new_data2_classes = []
	for cli in range(len(common_labels)):
		ccv1 = data1_class_labels.index(common_labels[cli])
		ccv_indices1 = [i for i, x in enumerate(data1_classes) if x == ccv1]
		ccv2 = data2_class_labels.index(common_labels[cli])
		ccv_indices2 = [i for i, x in enumerate(data2_classes) if x == ccv2]		
		for ccv_i in ccv_indices1:
			new_data1.append(data1[ccv_i])
			new_data1_classes.append(cli)
		for ccv_i in ccv_indices2:
			new_data2.append(data2[ccv_i])
			new_data2_classes.append(cli)
	return(new_data1, new_data1_classes, new_data2, new_data2_classes, common_labels)

def relabel_list(list_orig, mapping_list):		#mapping list is just a list of new labels where the index is the label in the orig list
	new_list = []
	for o_val in list_orig:
		new_list.append(mapping_list[o_val])
	return(new_list)

def table(lst, classes):
	result_table = []
	for l in lst:
		table_row = []
		for c in classes:
			if l == c:
				table_row.append(1.0)
			else:
				table_row.append(0.0)
		result_table.append(table_row)
	return(result_table)

def main():
	parser = optparse.OptionParser()
	parser.add_option("--in_data1", dest="in_data1", action="store", default="", help="")
	parser.add_option("--in_data2", dest="in_data2", action="store", default="", help="")
	parser.add_option("--in_clin1", dest="in_clin1", action="store", default="", help="")
	parser.add_option("--in_clin2", dest="in_clin2", action="store", default="", help="")
	parser.add_option("--attribute", dest="attribute", action="store", default="", help="")
	parser.add_option("--out_file_prefix", dest="out_file_prefix", action="store", default="", help="needs to include whole path")
	opts, args = parser.parse_args()
	#process input arguments:
	in_data1_file = opts.in_data1
	in_data2_file = opts.in_data2
	in_clin1_file = opts.in_clin1
	in_clin2_file = opts.in_clin2
	attribute = opts.attribute
	out_file_prefix = opts.out_file_prefix

	tmp = in_data1_file.split("/")
	in_data1_file_nopath = tmp[len(tmp) - 1]
	tmp = in_data2_file.split("/")
	in_data2_file_nopath = tmp[len(tmp) - 1]
	
	print >> sys.stderr, "Reading input matrices ..."
	(X1,Y1,Y1_str_labels,X1_features) = read_input(in_data1_file, in_clin1_file, attribute)
	(X2,Y2,Y2_str_labels,X2_features) = read_input(in_data2_file, in_clin2_file, attribute)

	#limit to common gene features:
	print >> sys.stderr, "Filtering on common genes ..."
	(X1_common_genes, X2_common_genes, common_features) = filter_on_common_genes(X1, X1_features, X2, X2_features)

	#filter out classes that have too few samples:
	print >> sys.stderr, "Filtering on class counts ..."
	(X1_filtered, Y1_filtered) = filter_on_class_counts(X1_common_genes, Y1, 3)
	(X2_filtered, Y2_filtered) = filter_on_class_counts(X2_common_genes, Y2, 3)

	#limit to common classes between the two datasets and re-map classes to the common nomenclature:
	print >> sys.stderr, "Filtering on common classes ..."
	(X1_final, Y1_final, X2_final, Y2_final, common_classes_map) = filter_on_common_classes(X1_filtered, Y1_filtered, Y1_str_labels, X2_filtered, Y2_filtered, Y2_str_labels)

	#now that we filtered the classes by class counts we need to update the list of class strings to reflect the filtering:
	new_Y1_filtered = []
	y1_filtered_classes = set(Y1_filtered)
	for y_i in range(len(Y1_str_labels)):
		if y_i in y1_filtered_classes:
			new_Y1_filtered.append(Y1_str_labels[y_i])

	Y1_str_labels = new_Y1_filtered

	new_Y2_filtered = []
	y2_filtered_classes = set(Y2_filtered)
	for y_i in range(len(Y2_str_labels)):
		if y_i in y2_filtered_classes:
			new_Y2_filtered.append(Y2_str_labels[y_i])

	Y2_str_labels = new_Y2_filtered
	
	for train_index, test_index in skf1:
		print("TRAIN:", train_index, "TEST:", test_index)	
	
	##########################################################################################################
	###file1 vs file2:
	##########################################################################################################

	print >> sys.stderr, "########################"
	print >> sys.stderr, "File1 vs file2"
	print >> sys.stderr, "########################"

	per_class_auc = {}
	for c in common_classes_map:
		per_class_auc[c] = {}
		per_class_auc[c]["classifier"] = []
		per_class_auc[c]["auc"] = []

	classifier_accuracy = {}
	classifier_accuracy["classifier"] = []
	classifier_accuracy["accuracy"] = []

	##########################################################################################################
	###file1 vs file2:
	print >> sys.stderr, "Building a linear model ..."
	svm = sklearn.svm.SVC(kernel="linear")
	svm_fit = svm.fit(X1_final,Y1_final)
	y_pred = svm_fit.predict(X2_final)
	y_pred_str = relabel_list(y_pred, common_classes_map)
	y_actual_str = relabel_list(Y2_final, common_classes_map)
	
	print >> sys.stderr, "\tOutputting results ..."
	#accuracy:
	accuracy = svm_fit.score(X2_final,Y2_final)
	classifier_accuracy["classifier"].append("svm linear")
	classifier_accuracy["accuracy"].append(accuracy)

	#open pdf for plot output:
	output_file_name_pdf = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".linear.pdf"
	pp = PdfPages(output_file_name_pdf)

	#confusion matrix:
	output_file_name_cm = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".cm.linear.tab"
	cm = confusion_matrix(y_actual_str, y_pred_str)
	actual_labels = []
	pred_labels = []
	for c in common_classes_map:
		actual_labels.append("actual."+c)
		pred_labels.append("pred."+c)
	output_file_cm = open(output_file_name_cm,'w')
	print >> output_file_cm, "label"+"\t"+"\t".join(pred_labels)
	for row_ind in range(len(cm)):
		print_row = list(cm[row_ind])
		print_row = [str(x) for x in print_row]
		print >> output_file_cm, actual_labels[row_ind]+"\t"+"\t".join(print_row)
	output_file_cm.close()
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	#plt.show()
	#plt.savefig(pp, format='pdf')
	pp.savefig()

	#roc/auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	Y2_table = table(Y2_final, range(len(common_classes_map)))
	Y2_table_t = zip(*Y2_table)
	y_score = svm_fit.decision_function(X2_final)
	for i in range(len(common_classes_map)):
		fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(Y2_table_t[i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#average auc and mean roc:
	ave_auc = sum([roc_auc[k] for k in roc_auc.keys()]) / float(len(roc_auc))
	ave_fpr = {"mean":[]}
	ave_tpr = {"mean":[]}
	for s in range(len(fpr[0])):
		fpr_s_total = 0.0
		tpr_s_total = 0.0
		for i in range(len(common_classes_map)):
			if (s < len(fpr[i])) and (s < len(tpr[i])):
				fpr_s_total += fpr[i][s]
				tpr_s_total += tpr[i][s]
		ave_fpr_s = fpr_s_total / len(common_classes_map)
		ave_tpr_s = tpr_s_total / len(common_classes_map)
		ave_fpr["mean"].append(ave_fpr_s)
		ave_tpr["mean"].append(ave_tpr_s)

	plt.figure()
	plt.plot(ave_fpr["mean"],ave_tpr["mean"], label='AUC = %0.3f' % ave_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	# Plot ROC curve
	plt.figure()
	for i in range(len(common_classes_map)):
		plt.plot(fpr[i], tpr[i], label=common_classes_map[i]+' (%0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Multi-class ROC')
	plt.legend(loc="lower right",fontsize=9)
	#plt.show()
	pp.savefig()

	pp.close()

	for i in range(len(common_classes_map)):
		per_class_auc[common_classes_map[i]]["classifier"].append('svm linear')
		per_class_auc[common_classes_map[i]]["auc"].append(roc_auc[i])

	##########################################################################################################
	###file1 vs file2:
	print >> sys.stderr, "Building a rbf model ..."
	svm = sklearn.svm.SVC(kernel="rbf")
	svm_fit = svm.fit(X1_final,Y1_final)
	y_pred = svm_fit.predict(X2_final)
	y_pred_str = relabel_list(y_pred, common_classes_map)
	y_actual_str = relabel_list(Y2_final, common_classes_map)
	
	print >> sys.stderr, "\tOutputting results ..."
	#accuracy:
	accuracy = svm_fit.score(X2_final,Y2_final)
	classifier_accuracy["classifier"].append("svm rbf")
	classifier_accuracy["accuracy"].append(accuracy)

	#open pdf for plot output:
	output_file_name_pdf = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".rbf.pdf"
	pp = PdfPages(output_file_name_pdf)

	#confusion matrix:
	output_file_name_cm = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".cm.rbf.tab"
	cm = confusion_matrix(y_actual_str, y_pred_str)
	actual_labels = []
	pred_labels = []
	for c in common_classes_map:
		actual_labels.append("actual."+c)
		pred_labels.append("pred."+c)
	output_file_cm = open(output_file_name_cm,'w')
	print >> output_file_cm, "label"+"\t"+"\t".join(pred_labels)
	for row_ind in range(len(cm)):
		print_row = list(cm[row_ind])
		print_row = [str(x) for x in print_row]
		print >> output_file_cm, actual_labels[row_ind]+"\t"+"\t".join(print_row)
	output_file_cm.close()
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	pp.savefig()

	#roc/auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	Y2_table = table(Y2_final, range(len(common_classes_map)))
	Y2_table_t = zip(*Y2_table)
	y_score = svm_fit.decision_function(X2_final)
	for i in range(len(common_classes_map)):
		fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(Y2_table_t[i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#average auc and mean roc:
	ave_auc = sum([roc_auc[k] for k in roc_auc.keys()]) / float(len(roc_auc))
	ave_fpr = {"mean":[]}
	ave_tpr = {"mean":[]}
	for s in range(len(fpr[0])):
		fpr_s_total = 0.0
		tpr_s_total = 0.0
		for i in range(len(common_classes_map)):
			if (s < len(fpr[i])) and (s < len(tpr[i])):
				fpr_s_total += fpr[i][s]
				tpr_s_total += tpr[i][s]
		ave_fpr_s = fpr_s_total / len(common_classes_map)
		ave_tpr_s = tpr_s_total / len(common_classes_map)
		ave_fpr["mean"].append(ave_fpr_s)
		ave_tpr["mean"].append(ave_tpr_s)

	plt.figure()
	plt.plot(ave_fpr["mean"],ave_tpr["mean"], label='AUC = %0.3f' % ave_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	# Plot ROC curve
	plt.figure()
	for i in range(len(common_classes_map)):
		plt.plot(fpr[i], tpr[i], label=common_classes_map[i]+' (%0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Multi-class ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	pp.close()

	for i in range(len(common_classes_map)):
		per_class_auc[common_classes_map[i]]["classifier"].append('svm rbf')
		per_class_auc[common_classes_map[i]]["auc"].append(roc_auc[i])

	##########################################################################################################
	###file1 vs file2:
	print >> sys.stderr, "Building a polynomial model ..."
	svm = sklearn.svm.SVC(kernel="poly")
	svm_fit = svm.fit(X1_final,Y1_final)
	y_pred = svm_fit.predict(X2_final)
	y_pred_str = relabel_list(y_pred, common_classes_map)
	y_actual_str = relabel_list(Y2_final, common_classes_map)

	print >> sys.stderr, "\tOutputting results ..."
	#accuracy:
	accuracy = svm_fit.score(X2_final,Y2_final)
	classifier_accuracy["classifier"].append("svm poly")
	classifier_accuracy["accuracy"].append(accuracy)

	#open pdf for plot output:
	output_file_name_pdf = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".poly.pdf"
	pp = PdfPages(output_file_name_pdf)

	#confusion matrix:
	output_file_name_cm = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".cm.poly.tab"
	cm = confusion_matrix(y_actual_str, y_pred_str)
	actual_labels = []
	pred_labels = []
	for c in common_classes_map:
		actual_labels.append("actual."+c)
		pred_labels.append("pred."+c)
	output_file_cm = open(output_file_name_cm,'w')
	print >> output_file_cm, "label"+"\t"+"\t".join(pred_labels)
	for row_ind in range(len(cm)):
		print_row = list(cm[row_ind])
		print_row = [str(x) for x in print_row]
		print >> output_file_cm, actual_labels[row_ind]+"\t"+"\t".join(print_row)
	output_file_cm.close()
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	pp.savefig()

	#roc/auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	Y2_table = table(Y2_final, range(len(common_classes_map)))
	Y2_table_t = zip(*Y2_table)
	y_score = svm_fit.decision_function(X2_final)
	for i in range(len(common_classes_map)):
		fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(Y2_table_t[i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#average auc and mean roc:
	ave_auc = sum([roc_auc[k] for k in roc_auc.keys()]) / float(len(roc_auc))
	ave_fpr = {"mean":[]}
	ave_tpr = {"mean":[]}
	for s in range(len(fpr[0])):
		fpr_s_total = 0.0
		tpr_s_total = 0.0
		for i in range(len(common_classes_map)):
			if (s < len(fpr[i])) and (s < len(tpr[i])):
				fpr_s_total += fpr[i][s]
				tpr_s_total += tpr[i][s]
		ave_fpr_s = fpr_s_total / len(common_classes_map)
		ave_tpr_s = tpr_s_total / len(common_classes_map)
		ave_fpr["mean"].append(ave_fpr_s)
		ave_tpr["mean"].append(ave_tpr_s)

	plt.figure()
	plt.plot(ave_fpr["mean"],ave_tpr["mean"], label='AUC = %0.3f' % ave_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	# Plot ROC curve
	plt.figure()
	for i in range(len(common_classes_map)):
		plt.plot(fpr[i], tpr[i], label=common_classes_map[i]+' (%0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Multi-class ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	pp.close()

	for i in range(len(common_classes_map)):
		per_class_auc[common_classes_map[i]]["classifier"].append('svm poly')
		per_class_auc[common_classes_map[i]]["auc"].append(roc_auc[i])

	##########################################################################################################
	###file1 vs file2:
	print >> sys.stderr, "Building a lda model ..."
	lda = sklearn.lda.LDA()
	lda_fit = lda.fit(X1_final,Y1_final,store_covariance=True)
	y_pred = lda_fit.predict(X2_final)
	y_pred_str = relabel_list(y_pred, common_classes_map)
	y_actual_str = relabel_list(Y2_final, common_classes_map)

	print >> sys.stderr, "\tOutputting results ..."
	#accuracy:
	accuracy = svm_fit.score(X2_final,Y2_final)
	classifier_accuracy["classifier"].append("lda")
	classifier_accuracy["accuracy"].append(accuracy)

	#open pdf for plot output:
	output_file_name_pdf = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".lda.pdf"
	pp = PdfPages(output_file_name_pdf)

	#confusion matrix:
	output_file_name_cm = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".cm.lda.tab"
	cm = confusion_matrix(y_actual_str, y_pred_str)
	actual_labels = []
	pred_labels = []
	for c in common_classes_map:
		actual_labels.append("actual."+c)
		pred_labels.append("pred."+c)
	output_file_cm = open(output_file_name_cm,'w')
	print >> output_file_cm, "label"+"\t"+"\t".join(pred_labels)
	for row_ind in range(len(cm)):
		print_row = list(cm[row_ind])
		print_row = [str(x) for x in print_row]
		print >> output_file_cm, actual_labels[row_ind]+"\t"+"\t".join(print_row)
	output_file_cm.close()
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	pp.savefig()

	#roc/auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	Y2_table = table(Y2_final, range(len(common_classes_map)))
	Y2_table_t = zip(*Y2_table)
	y_score = svm_fit.decision_function(X2_final)
	for i in range(len(common_classes_map)):
		fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(Y2_table_t[i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#average auc and mean roc:
	ave_auc = sum([roc_auc[k] for k in roc_auc.keys()]) / float(len(roc_auc))
	ave_fpr = {"mean":[]}
	ave_tpr = {"mean":[]}
	for s in range(len(fpr[0])):
		fpr_s_total = 0.0
		tpr_s_total = 0.0
		for i in range(len(common_classes_map)):
			if (s < len(fpr[i])) and (s < len(tpr[i])):
				fpr_s_total += fpr[i][s]
				tpr_s_total += tpr[i][s]
		ave_fpr_s = fpr_s_total / len(common_classes_map)
		ave_tpr_s = tpr_s_total / len(common_classes_map)
		ave_fpr["mean"].append(ave_fpr_s)
		ave_tpr["mean"].append(ave_tpr_s)

	plt.figure()
	plt.plot(ave_fpr["mean"],ave_tpr["mean"], label='AUC = %0.3f' % ave_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	# Plot ROC curve
	plt.figure()
	for i in range(len(common_classes_map)):
		plt.plot(fpr[i], tpr[i], label=common_classes_map[i]+' (%0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Multi-class ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	pp.close()

	for i in range(len(common_classes_map)):
		per_class_auc[common_classes_map[i]]["classifier"].append('lda')
		per_class_auc[common_classes_map[i]]["auc"].append(roc_auc[i])

	##########################################################################################################
	###file1 vs file2:
	print >> sys.stderr, "Building a NB model ..."
	gnb = sklearn.naive_bayes.GaussianNB()
	gnb_fit = gnb.fit(X1_final,Y1_final)
	y_pred = gnb_fit.predict(X2_final)
	y_pred_str = relabel_list(y_pred, common_classes_map)
	y_actual_str = relabel_list(Y2_final, common_classes_map)

	print >> sys.stderr, "\tOutputting results ..."
	#accuracy:
	accuracy = svm_fit.score(X2_final,Y2_final)
	classifier_accuracy["classifier"].append("naive bayes")
	classifier_accuracy["accuracy"].append(accuracy)

	#open pdf for plot output:
	output_file_name_pdf = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".nb.pdf"
	pp = PdfPages(output_file_name_pdf)

	#confusion matrix:
	output_file_name_cm = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".cm.nb.tab"
	cm = confusion_matrix(y_actual_str, y_pred_str)
	actual_labels = []
	pred_labels = []
	for c in common_classes_map:
		actual_labels.append("actual."+c)
		pred_labels.append("pred."+c)
	output_file_cm = open(output_file_name_cm,'w')
	print >> output_file_cm, "label"+"\t"+"\t".join(pred_labels)
	for row_ind in range(len(cm)):
		print_row = list(cm[row_ind])
		print_row = [str(x) for x in print_row]
		print >> output_file_cm, actual_labels[row_ind]+"\t"+"\t".join(print_row)
	output_file_cm.close()
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	pp.savefig()

	#roc/auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	Y2_table = table(Y2_final, range(len(common_classes_map)))
	Y2_table_t = zip(*Y2_table)
	y_score = svm_fit.decision_function(X2_final)
	for i in range(len(common_classes_map)):
		fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(Y2_table_t[i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#average auc and mean roc:
	ave_auc = sum([roc_auc[k] for k in roc_auc.keys()]) / float(len(roc_auc))
	ave_fpr = {"mean":[]}
	ave_tpr = {"mean":[]}
	for s in range(len(fpr[0])):
		fpr_s_total = 0.0
		tpr_s_total = 0.0
		for i in range(len(common_classes_map)):
			if (s < len(fpr[i])) and (s < len(tpr[i])):
				fpr_s_total += fpr[i][s]
				tpr_s_total += tpr[i][s]
		ave_fpr_s = fpr_s_total / len(common_classes_map)
		ave_tpr_s = tpr_s_total / len(common_classes_map)
		ave_fpr["mean"].append(ave_fpr_s)
		ave_tpr["mean"].append(ave_tpr_s)

	plt.figure()
	plt.plot(ave_fpr["mean"],ave_tpr["mean"], label='AUC = %0.3f' % ave_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	# Plot ROC curve
	plt.figure()
	for i in range(len(common_classes_map)):
		plt.plot(fpr[i], tpr[i], label=common_classes_map[i]+' (%0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Multi-class ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	pp.close()

	for i in range(len(common_classes_map)):
		per_class_auc[common_classes_map[i]]["classifier"].append('naive bayes')
		per_class_auc[common_classes_map[i]]["auc"].append(roc_auc[i])

	##########################################################################################################
	###file1 vs file2:
	print >> sys.stderr, "Building a RF model ..."
	forest = sklearn.ensemble.RandomForestClassifier()
	forest_fit = forest.fit(X1_final,Y1_final)
	y_pred = forest_fit.predict(X2_final)
	y_pred_str = relabel_list(y_pred, common_classes_map)
	y_actual_str = relabel_list(Y2_final, common_classes_map)

	print >> sys.stderr, "\tOutputting results ..."
	#accuracy:
	accuracy = svm_fit.score(X2_final,Y2_final)
	classifier_accuracy["classifier"].append("rf")
	classifier_accuracy["accuracy"].append(accuracy)

	#open pdf for plot output:
	output_file_name_pdf = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".rf.pdf"
	pp = PdfPages(output_file_name_pdf)

	#confusion matrix:
	output_file_name_cm = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".cm.rf.tab"
	cm = confusion_matrix(y_actual_str, y_pred_str)
	actual_labels = []
	pred_labels = []
	for c in common_classes_map:
		actual_labels.append("actual."+c)
		pred_labels.append("pred."+c)
	output_file_cm = open(output_file_name_cm,'w')
	print >> output_file_cm, "label"+"\t"+"\t".join(pred_labels)
	for row_ind in range(len(cm)):
		print_row = list(cm[row_ind])
		print_row = [str(x) for x in print_row]
		print >> output_file_cm, actual_labels[row_ind]+"\t"+"\t".join(print_row)
	output_file_cm.close()
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	pp.savefig()

	#roc/auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	Y2_table = table(Y2_final, range(len(common_classes_map)))
	Y2_table_t = zip(*Y2_table)
	y_score = svm_fit.decision_function(X2_final)
	for i in range(len(common_classes_map)):
		fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(Y2_table_t[i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#average auc and mean roc:
	ave_auc = sum([roc_auc[k] for k in roc_auc.keys()]) / float(len(roc_auc))
	ave_fpr = {"mean":[]}
	ave_tpr = {"mean":[]}
	for s in range(len(fpr[0])):
		fpr_s_total = 0.0
		tpr_s_total = 0.0
		for i in range(len(common_classes_map)):
			if (s < len(fpr[i])) and (s < len(tpr[i])):
				fpr_s_total += fpr[i][s]
				tpr_s_total += tpr[i][s]
		ave_fpr_s = fpr_s_total / len(common_classes_map)
		ave_tpr_s = tpr_s_total / len(common_classes_map)
		ave_fpr["mean"].append(ave_fpr_s)
		ave_tpr["mean"].append(ave_tpr_s)

	plt.figure()
	plt.plot(ave_fpr["mean"],ave_tpr["mean"], label='AUC = %0.3f' % ave_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	# Plot ROC curve
	plt.figure()
	for i in range(len(common_classes_map)):
		plt.plot(fpr[i], tpr[i], label=common_classes_map[i]+' (%0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Multi-class ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	pp.close()

	for i in range(len(common_classes_map)):
		per_class_auc[common_classes_map[i]]["classifier"].append('rf')
		per_class_auc[common_classes_map[i]]["auc"].append(roc_auc[i])

	##########################################################################################################
	###per class classifier summary:
	print >> sys.stderr, "Outputting summary results ..."
	output_file_name_auc = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".auc.tab"
	output_file_auc = open(output_file_name_auc,'w')
	classifiers = per_class_auc[per_class_auc.keys()[0]]["classifier"]
	print >> output_file_auc, "auc"+"\t"+"\t".join(classifiers)
	for i in range(len(common_classes_map)):
		output_line = common_classes_map[i]
		for c_i in range(len(classifiers)):
			output_line = output_line + "\t"+str(per_class_auc[common_classes_map[i]]["auc"][c_i])
		print >> output_file_auc, output_line
	output_file_auc.close()

	print >> sys.stderr, classifier_accuracy
	output_file_name_acc = out_file_prefix+in_data1_file_nopath+"__vs__"+in_data2_file_nopath+".accuracy.tab"
	output_file_acc = open(output_file_name_acc,'w')
	for a_i in range(len(classifier_accuracy["classifier"])):
		print >> output_file_acc, classifier_accuracy["classifier"][a_i]+"\t"+str(classifier_accuracy["accuracy"][a_i])
	output_file_acc.close()
	
	##########################################################################################################
	###file2 vs file1:
	##########################################################################################################

	print >> sys.stderr, "########################"
	print >> sys.stderr, "File2 vs file1"
	print >> sys.stderr, "########################"

	per_class_auc = {}
	for c in common_classes_map:
		per_class_auc[c] = {}
		per_class_auc[c]["classifier"] = []
		per_class_auc[c]["auc"] = []

	classifier_accuracy = {}
	classifier_accuracy["classifier"] = []
	classifier_accuracy["accuracy"] = []

	##########################################################################################################
	###file2 vs file1:
	print >> sys.stderr, "Building a linear model ..."
	svm = sklearn.svm.SVC(kernel="linear")
	svm_fit = svm.fit(X2_final,Y2_final)
	y_pred = svm_fit.predict(X1_final)
	y_pred_str = relabel_list(y_pred, common_classes_map)
	y_actual_str = relabel_list(Y1_final, common_classes_map)

	print >> sys.stderr, "\tOutputting results ..."
	#accuracy:
	accuracy = svm_fit.score(X1_final,Y1_final)
	classifier_accuracy["classifier"].append("svm linear")
	classifier_accuracy["accuracy"].append(accuracy)

	#open pdf for plot output:
	output_file_name_pdf = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".linear.pdf"
	pp = PdfPages(output_file_name_pdf)

	#confusion matrix:
	output_file_name_cm = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".cm.linear.tab"
	cm = confusion_matrix(y_actual_str, y_pred_str)
	actual_labels = []
	pred_labels = []
	for c in common_classes_map:
		actual_labels.append("actual."+c)
		pred_labels.append("pred."+c)
	output_file_cm = open(output_file_name_cm,'w')
	print >> output_file_cm, "label"+"\t"+"\t".join(pred_labels)
	for row_ind in range(len(cm)):
		print_row = list(cm[row_ind])
		print_row = [str(x) for x in print_row]
		print >> output_file_cm, actual_labels[row_ind]+"\t"+"\t".join(print_row)
	output_file_cm.close()
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	#plt.show()
	#plt.savefig(pp, format='pdf')
	pp.savefig()

	#roc/auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	Y1_table = table(Y1_final, range(len(common_classes_map)))
	Y1_table_t = zip(*Y1_table)
	y_score = svm_fit.decision_function(X1_final)
	for i in range(len(common_classes_map)):
		fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(Y1_table_t[i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#average auc and mean roc:
	ave_auc = sum([roc_auc[k] for k in roc_auc.keys()]) / float(len(roc_auc))
	ave_fpr = {"mean":[]}
	ave_tpr = {"mean":[]}
	for s in range(len(fpr[0])):
		fpr_s_total = 0.0
		tpr_s_total = 0.0
		for i in range(len(common_classes_map)):
			if (s < len(fpr[i])) and (s < len(tpr[i])):
				fpr_s_total += fpr[i][s]
				tpr_s_total += tpr[i][s]
		ave_fpr_s = fpr_s_total / len(common_classes_map)
		ave_tpr_s = tpr_s_total / len(common_classes_map)
		ave_fpr["mean"].append(ave_fpr_s)
		ave_tpr["mean"].append(ave_tpr_s)

	plt.figure()
	plt.plot(ave_fpr["mean"],ave_tpr["mean"], label='AUC = %0.3f' % ave_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	# Plot ROC curve
	plt.figure()
	for i in range(len(common_classes_map)):
		plt.plot(fpr[i], tpr[i], label=common_classes_map[i]+' (%0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Multi-class ROC')
	plt.legend(loc="lower right",fontsize=9)
	#plt.show()
	pp.savefig()

	pp.close()

	for i in range(len(common_classes_map)):
		per_class_auc[common_classes_map[i]]["classifier"].append('svm linear')
		per_class_auc[common_classes_map[i]]["auc"].append(roc_auc[i])

	##########################################################################################################
	###file2 vs file1:
	print >> sys.stderr, "Building a rbf model ..."
	svm = sklearn.svm.SVC(kernel="rbf")
	svm_fit = svm.fit(X2_final,Y2_final)
	y_pred = svm_fit.predict(X1_final)
	y_pred_str = relabel_list(y_pred, common_classes_map)
	y_actual_str = relabel_list(Y1_final, common_classes_map)
	
	print >> sys.stderr, "\tOutputting results ..."
	#accuracy:
	accuracy = svm_fit.score(X1_final,Y1_final)
	classifier_accuracy["classifier"].append("svm rbf")
	classifier_accuracy["accuracy"].append(accuracy)

	#open pdf for plot output:
	output_file_name_pdf = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".rbf.pdf"
	pp = PdfPages(output_file_name_pdf)

	#confusion matrix:
	output_file_name_cm = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".cm.rbf.tab"
	cm = confusion_matrix(y_actual_str, y_pred_str)
	actual_labels = []
	pred_labels = []
	for c in common_classes_map:
		actual_labels.append("actual."+c)
		pred_labels.append("pred."+c)
	output_file_cm = open(output_file_name_cm,'w')
	print >> output_file_cm, "label"+"\t"+"\t".join(pred_labels)
	for row_ind in range(len(cm)):
		print_row = list(cm[row_ind])
		print_row = [str(x) for x in print_row]
		print >> output_file_cm, actual_labels[row_ind]+"\t"+"\t".join(print_row)
	output_file_cm.close()
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	pp.savefig()

	#roc/auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	Y1_table = table(Y1_final, range(len(common_classes_map)))
	Y1_table_t = zip(*Y1_table)
	y_score = svm_fit.decision_function(X1_final)
	for i in range(len(common_classes_map)):
		fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(Y1_table_t[i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#average auc and mean roc:
	ave_auc = sum([roc_auc[k] for k in roc_auc.keys()]) / float(len(roc_auc))
	ave_fpr = {"mean":[]}
	ave_tpr = {"mean":[]}
	for s in range(len(fpr[0])):
		fpr_s_total = 0.0
		tpr_s_total = 0.0
		for i in range(len(common_classes_map)):
			if (s < len(fpr[i])) and (s < len(tpr[i])):
				fpr_s_total += fpr[i][s]
				tpr_s_total += tpr[i][s]
		ave_fpr_s = fpr_s_total / len(common_classes_map)
		ave_tpr_s = tpr_s_total / len(common_classes_map)
		ave_fpr["mean"].append(ave_fpr_s)
		ave_tpr["mean"].append(ave_tpr_s)

	plt.figure()
	plt.plot(ave_fpr["mean"],ave_tpr["mean"], label='AUC = %0.3f' % ave_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	# Plot ROC curve
	plt.figure()
	for i in range(len(common_classes_map)):
		plt.plot(fpr[i], tpr[i], label=common_classes_map[i]+' (%0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Multi-class ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	pp.close()

	for i in range(len(common_classes_map)):
		per_class_auc[common_classes_map[i]]["classifier"].append('svm rbf')
		per_class_auc[common_classes_map[i]]["auc"].append(roc_auc[i])

	##########################################################################################################
	###file2 vs file1:
	print >> sys.stderr, "Building a polynomial model ..."
	svm = sklearn.svm.SVC(kernel="poly")
	svm_fit = svm.fit(X2_final,Y2_final)
	y_pred = svm_fit.predict(X1_final)
	y_pred_str = relabel_list(y_pred, common_classes_map)
	y_actual_str = relabel_list(Y1_final, common_classes_map)

	print >> sys.stderr, "\tOutputting results ..."
	#accuracy:
	accuracy = svm_fit.score(X1_final,Y1_final)
	classifier_accuracy["classifier"].append("svm poly")
	classifier_accuracy["accuracy"].append(accuracy)

	#open pdf for plot output:
	output_file_name_pdf = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".poly.pdf"
	pp = PdfPages(output_file_name_pdf)

	#confusion matrix:
	output_file_name_cm = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".cm.poly.tab"
	cm = confusion_matrix(y_actual_str, y_pred_str)
	actual_labels = []
	pred_labels = []
	for c in common_classes_map:
		actual_labels.append("actual."+c)
		pred_labels.append("pred."+c)
	output_file_cm = open(output_file_name_cm,'w')
	print >> output_file_cm, "label"+"\t"+"\t".join(pred_labels)
	for row_ind in range(len(cm)):
		print_row = list(cm[row_ind])
		print_row = [str(x) for x in print_row]
		print >> output_file_cm, actual_labels[row_ind]+"\t"+"\t".join(print_row)
	output_file_cm.close()
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	pp.savefig()

	#roc/auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	Y1_table = table(Y1_final, range(len(common_classes_map)))
	Y1_table_t = zip(*Y1_table)
	y_score = svm_fit.decision_function(X1_final)
	for i in range(len(common_classes_map)):
		fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(Y1_table_t[i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#average auc and mean roc:
	ave_auc = sum([roc_auc[k] for k in roc_auc.keys()]) / float(len(roc_auc))
	ave_fpr = {"mean":[]}
	ave_tpr = {"mean":[]}
	for s in range(len(fpr[0])):
		fpr_s_total = 0.0
		tpr_s_total = 0.0
		for i in range(len(common_classes_map)):
			if (s < len(fpr[i])) and (s < len(tpr[i])):
				fpr_s_total += fpr[i][s]
				tpr_s_total += tpr[i][s]
		ave_fpr_s = fpr_s_total / len(common_classes_map)
		ave_tpr_s = tpr_s_total / len(common_classes_map)
		ave_fpr["mean"].append(ave_fpr_s)
		ave_tpr["mean"].append(ave_tpr_s)

	plt.figure()
	plt.plot(ave_fpr["mean"],ave_tpr["mean"], label='AUC = %0.3f' % ave_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	# Plot ROC curve
	plt.figure()
	for i in range(len(common_classes_map)):
		plt.plot(fpr[i], tpr[i], label=common_classes_map[i]+' (%0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Multi-class ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	pp.close()

	for i in range(len(common_classes_map)):
		per_class_auc[common_classes_map[i]]["classifier"].append('svm poly')
		per_class_auc[common_classes_map[i]]["auc"].append(roc_auc[i])

	##########################################################################################################
	###file2 vs file1:
	print >> sys.stderr, "Building a lda model ..."
	lda = sklearn.lda.LDA()
	lda_fit = lda.fit(X1_final,Y1_final,store_covariance=True)
	y_pred = lda_fit.predict(X1_final)
	y_pred_str = relabel_list(y_pred, common_classes_map)
	y_actual_str = relabel_list(Y1_final, common_classes_map)

	print >> sys.stderr, "\tOutputting results ..."
	#accuracy:
	accuracy = svm_fit.score(X1_final,Y1_final)
	classifier_accuracy["classifier"].append("lda")
	classifier_accuracy["accuracy"].append(accuracy)

	#open pdf for plot output:
	output_file_name_pdf = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".lda.pdf"
	pp = PdfPages(output_file_name_pdf)

	#confusion matrix:
	output_file_name_cm = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".cm.lda.tab"
	cm = confusion_matrix(y_actual_str, y_pred_str)
	actual_labels = []
	pred_labels = []
	for c in common_classes_map:
		actual_labels.append("actual."+c)
		pred_labels.append("pred."+c)
	output_file_cm = open(output_file_name_cm,'w')
	print >> output_file_cm, "label"+"\t"+"\t".join(pred_labels)
	for row_ind in range(len(cm)):
		print_row = list(cm[row_ind])
		print_row = [str(x) for x in print_row]
		print >> output_file_cm, actual_labels[row_ind]+"\t"+"\t".join(print_row)
	output_file_cm.close()
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	pp.savefig()

	#roc/auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	Y1_table = table(Y1_final, range(len(common_classes_map)))
	Y1_table_t = zip(*Y1_table)
	y_score = svm_fit.decision_function(X1_final)
	for i in range(len(common_classes_map)):
		fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(Y1_table_t[i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#average auc and mean roc:
	ave_auc = sum([roc_auc[k] for k in roc_auc.keys()]) / float(len(roc_auc))
	ave_fpr = {"mean":[]}
	ave_tpr = {"mean":[]}
	for s in range(len(fpr[0])):
		fpr_s_total = 0.0
		tpr_s_total = 0.0
		for i in range(len(common_classes_map)):
			if (s < len(fpr[i])) and (s < len(tpr[i])):
				fpr_s_total += fpr[i][s]
				tpr_s_total += tpr[i][s]
		ave_fpr_s = fpr_s_total / len(common_classes_map)
		ave_tpr_s = tpr_s_total / len(common_classes_map)
		ave_fpr["mean"].append(ave_fpr_s)
		ave_tpr["mean"].append(ave_tpr_s)

	plt.figure()
	plt.plot(ave_fpr["mean"],ave_tpr["mean"], label='AUC = %0.3f' % ave_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	# Plot ROC curve
	plt.figure()
	for i in range(len(common_classes_map)):
		plt.plot(fpr[i], tpr[i], label=common_classes_map[i]+' (%0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Multi-class ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	pp.close()

	for i in range(len(common_classes_map)):
		per_class_auc[common_classes_map[i]]["classifier"].append('lda')
		per_class_auc[common_classes_map[i]]["auc"].append(roc_auc[i])

	##########################################################################################################
	###file2 vs file1:
	print >> sys.stderr, "Building a NB model ..."
	gnb = sklearn.naive_bayes.GaussianNB()
	gnb_fit = gnb.fit(X2_final,Y2_final)
	y_pred = gnb_fit.predict(X1_final)
	y_pred_str = relabel_list(y_pred, common_classes_map)
	y_actual_str = relabel_list(Y1_final, common_classes_map)

	print >> sys.stderr, "\tOutputting results ..."
	#accuracy:
	accuracy = svm_fit.score(X1_final,Y1_final)
	classifier_accuracy["classifier"].append("naive bayes")
	classifier_accuracy["accuracy"].append(accuracy)

	#open pdf for plot output:
	output_file_name_pdf = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".nb.pdf"
	pp = PdfPages(output_file_name_pdf)

	#confusion matrix:
	output_file_name_cm = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".cm.nb.tab"
	cm = confusion_matrix(y_actual_str, y_pred_str)
	actual_labels = []
	pred_labels = []
	for c in common_classes_map:
		actual_labels.append("actual."+c)
		pred_labels.append("pred."+c)
	output_file_cm = open(output_file_name_cm,'w')
	print >> output_file_cm, "label"+"\t"+"\t".join(pred_labels)
	for row_ind in range(len(cm)):
		print_row = list(cm[row_ind])
		print_row = [str(x) for x in print_row]
		print >> output_file_cm, actual_labels[row_ind]+"\t"+"\t".join(print_row)
	output_file_cm.close()
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	pp.savefig()

	#roc/auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	Y1_table = table(Y1_final, range(len(common_classes_map)))
	Y1_table_t = zip(*Y1_table)
	y_score = svm_fit.decision_function(X1_final)
	for i in range(len(common_classes_map)):
		fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(Y1_table_t[i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#average auc and mean roc:
	ave_auc = sum([roc_auc[k] for k in roc_auc.keys()]) / float(len(roc_auc))
	ave_fpr = {"mean":[]}
	ave_tpr = {"mean":[]}
	for s in range(len(fpr[0])):
		fpr_s_total = 0.0
		tpr_s_total = 0.0
		for i in range(len(common_classes_map)):
			if (s < len(fpr[i])) and (s < len(tpr[i])):
				fpr_s_total += fpr[i][s]
				tpr_s_total += tpr[i][s]
		ave_fpr_s = fpr_s_total / len(common_classes_map)
		ave_tpr_s = tpr_s_total / len(common_classes_map)
		ave_fpr["mean"].append(ave_fpr_s)
		ave_tpr["mean"].append(ave_tpr_s)

	plt.figure()
	plt.plot(ave_fpr["mean"],ave_tpr["mean"], label='AUC = %0.3f' % ave_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	# Plot ROC curve
	plt.figure()
	for i in range(len(common_classes_map)):
		plt.plot(fpr[i], tpr[i], label=common_classes_map[i]+' (%0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Multi-class ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	pp.close()

	for i in range(len(common_classes_map)):
		per_class_auc[common_classes_map[i]]["classifier"].append('naive bayes')
		per_class_auc[common_classes_map[i]]["auc"].append(roc_auc[i])

	##########################################################################################################
	###file2 vs file1:
	print >> sys.stderr, "Building a RF model ..."
	forest = sklearn.ensemble.RandomForestClassifier()
	forest_fit = forest.fit(X2_final,Y2_final)
	y_pred = forest_fit.predict(X1_final)
	y_pred_str = relabel_list(y_pred, common_classes_map)
	y_actual_str = relabel_list(Y1_final, common_classes_map)

	print >> sys.stderr, "\tOutputting results ..."
	#accuracy:
	accuracy = svm_fit.score(X1_final,Y1_final)
	classifier_accuracy["classifier"].append("rf")
	classifier_accuracy["accuracy"].append(accuracy)

	#open pdf for plot output:
	output_file_name_pdf = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".rf.pdf"
	pp = PdfPages(output_file_name_pdf)

	#confusion matrix:
	output_file_name_cm = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".cm.rf.tab"
	cm = confusion_matrix(y_actual_str, y_pred_str)
	actual_labels = []
	pred_labels = []
	for c in common_classes_map:
		actual_labels.append("actual."+c)
		pred_labels.append("pred."+c)
	output_file_cm = open(output_file_name_cm,'w')
	print >> output_file_cm, "label"+"\t"+"\t".join(pred_labels)
	for row_ind in range(len(cm)):
		print_row = list(cm[row_ind])
		print_row = [str(x) for x in print_row]
		print >> output_file_cm, actual_labels[row_ind]+"\t"+"\t".join(print_row)
	output_file_cm.close()
	plt.matshow(cm)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	pp.savefig()

	#roc/auc
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	Y1_table = table(Y1_final, range(len(common_classes_map)))
	Y1_table_t = zip(*Y1_table)
	y_score = svm_fit.decision_function(X1_final)
	for i in range(len(common_classes_map)):
		fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(Y1_table_t[i], y_score[:, i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	#average auc and mean roc:
	ave_auc = sum([roc_auc[k] for k in roc_auc.keys()]) / float(len(roc_auc))
	ave_fpr = {"mean":[]}
	ave_tpr = {"mean":[]}
	for s in range(len(fpr[0])):
		fpr_s_total = 0.0
		tpr_s_total = 0.0
		for i in range(len(common_classes_map)):
			if (s < len(fpr[i])) and (s < len(tpr[i])):
				fpr_s_total += fpr[i][s]
				tpr_s_total += tpr[i][s]
		ave_fpr_s = fpr_s_total / len(common_classes_map)
		ave_tpr_s = tpr_s_total / len(common_classes_map)
		ave_fpr["mean"].append(ave_fpr_s)
		ave_tpr["mean"].append(ave_tpr_s)

	plt.figure()
	plt.plot(ave_fpr["mean"],ave_tpr["mean"], label='AUC = %0.3f' % ave_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	# Plot ROC curve
	plt.figure()
	for i in range(len(common_classes_map)):
		plt.plot(fpr[i], tpr[i], label=common_classes_map[i]+' (%0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Multi-class ROC')
	plt.legend(loc="lower right",fontsize=9)
	pp.savefig()

	pp.close()

	for i in range(len(common_classes_map)):
		per_class_auc[common_classes_map[i]]["classifier"].append('rf')
		per_class_auc[common_classes_map[i]]["auc"].append(roc_auc[i])

	##########################################################################################################
	###per class classifier summary:
	print >> sys.stderr, "Outputting summary results ..."
	output_file_name_auc = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".auc.tab"
	output_file_auc = open(output_file_name_auc,'w')
	classifiers = per_class_auc[per_class_auc.keys()[0]]["classifier"]
	print >> output_file_auc, "auc"+"\t"+"\t".join(classifiers)
	for i in range(len(common_classes_map)):
		output_line = common_classes_map[i]
		for c_i in range(len(classifiers)):
			output_line = output_line + "\t"+str(per_class_auc[common_classes_map[i]]["auc"][c_i])
		print >> output_file_auc, output_line
	output_file_auc.close()
	
	print >> sys.stderr, classifier_accuracy
	output_file_name_acc = out_file_prefix+in_data2_file_nopath+"__vs__"+in_data1_file_nopath+".accuracy.tab"
	output_file_acc = open(output_file_name_acc,'w')
	for a_i in range(len(classifier_accuracy["classifier"])):
		print >> output_file_acc, classifier_accuracy["classifier"][a_i]+"\t"+str(classifier_accuracy["accuracy"][a_i])
	output_file_acc.close()	

main()
