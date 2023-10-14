import os
import sys
import random
import math
import csv
import datetime
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

# 1. Choose the number of layers, 1, 2 or 3
# For base layer
# 	1. Generate a basic ascending, decending or static sequence, such as 2, 3, 4, 5.
# 	2. Choose a power for the numers, i.e. 2,3,4,5 or 2^2, 3^2, 4^2, or 2^3, 3^3, 4^3 etc.
# 	3. Choose a basic operation sequence for the numbers, i.e. +1, +2, +3, +4 or +1, -2, /3, *4 etc.
# For next layer
#	4. Generate a suitable random number to begin sequence
#	5. Complete sequence based on operations in previous layer
# 	6. Choose a power for the numers, i.e. 2,3,4,5 or 2^2, 3^2, 4^2, or 2^3, 3^3, 4^3 etc.
# 	7. Choose a basic operation sequence for the numbers, i.e. +1, +2, +3, +4 or +1, -2, /3, *4 etc.
# For any remaining layers
#	8. Repeat steps 4 - 7
# 9. Finish by generating a random starting number and applying last layer

# Number of layers to generate
num_layers = 2
# Number of Problems to generate
num_problems = 200

# Probability of picking a power ^2 or ^3 1 in pow_prob
pow_prob = 6

operations = {"Add":1,"Subtract":2,"Multiply":4,"Divide":5}

max_layers = 2

used_pow = False

used_OpSeq = []

glo_layer_count = 0
loc_layer_count = 0

val_num_scale = 20
# Maximum number in the base sequence of numbers, must be 7 or greater
op_num_scale = 7

location = "./"

diffscore = 0
gen_history = []

############################################################################################
# Increasing Sequence
# Create a basic increasing sequence of num values between the min and max values specified. 
############################################################################################
def IncSequence(min,max,num):
	
	sequence = []
	max_step = int(math.floor((max-min)/num))
	#print("max_step = "+str(max_step))
	if max_step > 1:
		step = random.randint(1,max_step)
	else:
		step = 1
	start = random.randint(min,max-(step*num))

	for i in range(0,num):
		start = start + step
		sequence.append(start)

	return sequence

############################################################################################
# Decreasing Sequence
# Create a basic decreasing sequence of num values between the min and max values specified. 
############################################################################################
def DecSequence(min,max,num):

	sequence = []
	max_step = int(math.floor((max-min)/num))
	#print("max_step = "+str(max_step))
	if max_step > 1:
		step = random.randint(1,max_step)
	else:
		step = 1
	start = random.randint(min+(step*num),max)

	for i in range(0,num):
		start = start - step
		sequence.append(start)

	return sequence

############################################################################################
# Flat Operation Sequence
# Create a sequence of num # operations values using only a single operation. 
############################################################################################
def FlatOpSequence(num):
	global diffscore

	sequence = []
	op = random.choice(list(operations.keys()))
	for i in range(0,num):
		sequence.append(op)

	diffscore = diffscore + operations[op]

	return sequence

############################################################################################
# Alternating Operation Sequence
# Create an alternating sequence of num # operations values using only two operations. 
############################################################################################
def AltOpSequence(num):
	global diffscore
	max_diff = 0

	sequence = []
	operations_copy = list(operations.keys())
	op1 = random.choice(operations_copy)
	operations_copy.remove(op1)
	op2 = random.choice(operations_copy)

	for i in range(1,num+1):
		if i%2 > 0:
			sequence.append(op1)
		else:
			sequence.append(op2)

	max_diff = max(operations[op1],operations[op2])
	diffscore = diffscore + 4 + max_diff

	return sequence

############################################################################################
# Pattern Operation Sequence
# Create a repeating pattern of num # operations values using all operations. 
############################################################################################
def PatOpSequence(num):
	global diffscore

	sequence = []
	operations_copy = list(operations.keys())
	ops = []
	for i in range(0,num-1):
		ops.append("")
		ops[i] = random.choice(operations_copy)
		operations_copy.remove(ops[i])

	ops.append(ops[0])

	for i in range(0,num):
		sequence.append(ops[i])

	diffscore = 12 + max_diff
	return sequence


############################################################################################
# Operation Sequence
# Create a basic sequence of operations values of length lim. 
############################################################################################
def OpSequence(num,max = 3):

	#print("OpSequence maximum = "+str(max))
	sequence = []
	poss = False

	if max == 1:
		ran = 1
	else:
		ran = random.randint(1,max)

	for i in range(1,max+1):
		if i not in used_OpSeq:
			poss = True

	if not poss or (len(used_OpSeq) > 2):
		return -1

	while ran in used_OpSeq and poss:
		ran = random.randint(1,max)

	if ran == 1:
		# flat sequence
		sequence = FlatOpSequence(num)
	if ran == 2:
		# alternating sequence
		sequence = AltOpSequence(num)
	if ran == 3:
		# patern sequence
		sequence = PatOpSequence(num)

	used_OpSeq.append(ran)

	return sequence

############################################################################################
# Calculate Values
# Calculate a layer of values for the problem, based on the previous layer of ops, step 5-7.
############################################################################################
def CalculateVals(PrevLayer,start):

	seq_num = float(start)
	sequence = []
	sequence.append(seq_num)
	#print("operations = "+str(PrevLayer[0]))
	#print("vals = "+str(PrevLayer[1]))

	for i in range(0,len(PrevLayer[0])):
		op = PrevLayer[0][i]
		factor = PrevLayer[1][i]
		#print(str(op)+" by "+str(factor))
		if op == "Add":
			seq_num = seq_num + factor
		if op == "Subtract":
			seq_num = seq_num - factor
		if op == "Multiply":
			seq_num = seq_num * factor
		if op == "Divide":
			if factor == 0:
				return -1
			else:
				seq_num = seq_num / factor

		seq_num = round(seq_num,5)
		sequence.append(seq_num)

		#print(seq_num)
	#print(sequence)
	return sequence



############################################################################################
# Generate Op Layer
# Generate a layer of operations values for the problem. As steps 1-3 or 5-7. 
############################################################################################
def GenerateOpLayer(PrevLayer):

	global pow_prob
	global loc_layer_count
	global used_pow
	global diffscore

	num_per_seq = 5
	ran_start = random.randint(1,val_num_scale)
	
	flat = False

	if len(PrevLayer) == 0:

		ran = random.randint(1,3)
		if ran == 1:
			num_seq = IncSequence(2,op_num_scale,num_per_seq)
			diffscore = diffscore + 2
		elif ran == 2:
			num_seq = DecSequence(2,op_num_scale,num_per_seq)
			diffscore = diffscore + 2
		else:
			num_seq = []
			ran_num = random.randint(2,op_num_scale)
			for i in range(0,num_per_seq):
				num_seq.append(ran_num)
			flat = True
	else:
		num_per_seq = len(PrevLayer[0])+1
		num_seq = CalculateVals(PrevLayer,ran_start)

		if num_seq == -1:
			return -1

	uni_nums = []
	for num in num_seq:
		if num not in uni_nums:
			uni_nums.append(num)

	op_seq = 3
	power = 1

	if not flat and not used_pow and len(uni_nums) > 2:
		power = random.randint(1,pow_prob)
		if power == pow_prob:
			print("raising to power 3")
			power = 3
			diffscore = diffscore + 12
			used_pow = True
		elif power == pow_prob-1:
			print("raising to power 2")
			power = 2
			diffscore = diffscore + 8
			used_pow = True
		else:
			power = 1

		for i in range(0,len(num_seq)):
			num_seq[i] = num_seq[i] ** power

		if power > 1:
			op_seq = 1

	op_sequence = OpSequence(num_per_seq,min(op_seq,max_layers-loc_layer_count))
	#print(op_sequence)
	#print(num_seq)

	loc_layer_count = loc_layer_count + 1

	return [op_sequence,num_seq]


############################################################################################
# Linear Extraction
# Take a sequence and predict the last value based on a linear regression of all previous points, to be used as an incorrect answer.
############################################################################################
def linearExtraction(sequence):
	
	y = pd.DataFrame(sequence[:-1])

	lm = linear_model.LinearRegression()
	axis = []
	
	for i in range(0,len(sequence)-1):
		axis.append(i)
	
	x = pd.DataFrame(axis)
	model = lm.fit(x,y)
	pred = lm.predict(np.array([len(sequence)-1]).reshape(1, -1))

	pred = round(pred[0][0],5)

	return pred


############################################################################################
# Polynomial Extraction
# Take a sequence and an order of polynomial to fit to all but that last value in the sequence and generate a fake final answer.
############################################################################################
def polyExtraction(sequence,order):
	
	axis = []
	
	for i in range(0,len(sequence)-1):
		axis.append(i)

	x = pd.DataFrame(axis)
	x_pred = pd.DataFrame([len(sequence)-1])
	
	y = pd.DataFrame(sequence[:-1])
	
	#degree = "degree = "+str(order)

	quadratic_featurizer = PolynomialFeatures(degree = order) 
	
	x_quad = quadratic_featurizer.fit_transform(x)
	xp_quad = quadratic_featurizer.fit_transform(x_pred)
	
	lm = linear_model.LinearRegression() 
	lm.fit(x_quad, y)
	
	pred = lm.predict(xp_quad) 
	#print(pred)

	pred = round(pred[0][0],5)

	return pred

############################################################################################
# Generate Fake Answers
# Take the correct sequence and generate four 'fake' possible answers by using various sequence extending techniques.
############################################################################################
def GenerateFakeAnswers(sequence):
	lin = linearExtraction(sequence)
	quad = polyExtraction(sequence,2)
	cubic = polyExtraction(sequence,3)
	avg = (lin + quad + cubic) /3

	avg = round(avg,5)

	fakes = [lin,quad,cubic,avg]

	return fakes

############################################################################################
# Get number of decimal places
# As we will display all numbers together as text, it is important the string values of the sequence match the fake answers.
# This function calculates the number of decimal places used in the string representaion of correct sequence answer.
############################################################################################
def getDec(num):
	num_str = str(num)
	#print("num_str: "+num_str)
	num_str = num_str.lower()
	count = 0
	trail = ''

	if ('e' in num_str):
		#print("getDec returning -1")
		return -1
	else:
		pos = num_str.find('.')
		trail = num_str[pos:]
		#print("trail: "+trail)
		for i in range(1,len(trail)+1):
			if trail[-i] == '0':
				count = count + 1
			else:
				break
	#print("count = "+str(count))
	#print("getDec returning "+str(len(trail)-count-1))

	return len(trail)-count-1

############################################################################################
# Change number of decimal places
# This reduces the number of decimal places which appear in the fake answer to match that of the correct answer.
############################################################################################
def changeDec(num,ans_dec):
	num_str = str(num)

	if 'e' in num_str:
		return False

	else:
		pos = num_str.find('.')

		num_str = num_str[:pos+ans_dec]

		return float(num_str)

def writeToCSV(itt,sequence,fake_ans,path):
	
	file_loc = path + "NumSequence.csv"

	exists = os.path.isfile(file_loc)

	if not exists:
		csvfile = open(file_loc, 'w')
		csvfile.write("Difficulty,Unique ID,Seq 1,Seq 2,Seq 3,Seq 4,Seq 5,Seq 6,,Ans 1,Ans 2,Ans 3,Ans 4,Ans 5,,Correct Answer")
		csvfile.write("\n")
		csvfile.close()

	csvfile = open(file_loc, 'a')
	#filewriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
	
	csvfile.write(str(diffscore) + "," + str(itt)+ ",")

	for i in range(0,len(sequence[:-1])):
		csvfile.write(str(sequence[i]) + ",")
	
	if (len(sequence[:-1]) < 6):
		for i in range(0,6-len(sequence[:-1])):
			csvfile.write(",")


	csvfile.write(",")
	
	for i in range(0,len(fake_ans)):
		csvfile.write(str(fake_ans[i]) + ",")

	csvfile.write(",")
	csvfile.write(str(sequence[-1]))


	csvfile.write("\n")
	csvfile.close()

############################################################################################
# Calculate Difficulty
# If testing is not available, estimate a difficulty based on the operations used to generate the sequence.
############################################################################################
def CalcDifficulty(diff_array,sequence):
	global diffscore
	diff = 0
	neg = False
	max_dec = -999

	for i in range(len(diff_array)):
		diff = diff + diff_array[i]

	# Check if sequence contains unique numbers to estimate difficulty. Multiple operations may have cancelled one another.
	uni_nums = []
	for num in sequence:
		if num not in uni_nums:
			uni_nums.append(num)

	for num in sequence:
		if num < 0:
			neg = True
		max_dec = max(max_dec,getDec(num))

	if neg:
		diff = diff + 2

	# If many decimals, it is likely more difficult for the candidate.
	if max_dec > 1:
		diff = diff + 3
	if max_dec > 2:
		diff = diff + 2



	if len(uni_nums) < len(sequence[:-1]):
		if len(uni_nums) == 1:
			diff = 0
		if len(uni_nums) == 2:
			diff = 2
		if len(uni_nums) > 3:
			diff = 4

	return diff


############################################################################################
# Generate A Sequence Set
# Using while loops and random number generation, create a set of sequences that all conform to the item criteria, skipping the iteration where unsuitable results are produced.
############################################################################################
def GenerateSequenceSet(itt,num_layers,path):

	global glo_layer_count
	global used_pow
	global used_OpSeq
	global diffscore
	global loc_layer_count

	added = False
	while not added:
		fail = True
		max_tries = 100
		count = 0

		while fail:
			diffscore = 0
			loc_layer_count = int(glo_layer_count)
			used_pow = False
			used_OpSeq = []

			fail = False
			count = count + 1
			
			targetDiff = 8
			OpLayer = []
			layers = 0
			diff_array = []

			for i in range(0,num_layers):
				if i < max_layers:
					new_OpLayer = GenerateOpLayer(OpLayer)
					if new_OpLayer == -1:
						print("Cannot produce "+str(num_layers)+" layers")
						break
					if new_OpLayer[0] == -1:
						print("Cannot produce "+str(num_layers)+" layers")
						break
					else:
						print("Layer Ops = "+str(new_OpLayer[0]))
						print("Layer Vals = "+str(new_OpLayer[1]))
						OpLayer = new_OpLayer
					layers = layers + 1

			ran_start = random.randint(1,val_num_scale)
			sequence = CalculateVals(OpLayer,ran_start)
			if sequence == -1:
				fail = True
				continue

			# Check if answer is not the same as the previous value
			if sequence[-2] == sequence[-1]:
				fail = True
				continue

			min_num = 999

			# # Check all numbers are > than some min 0.001 maybe
			# for i in range(0,len(sequence)):
			# 	ans_dec = getDec(sequence[i])
			# 	if ans_dec > 5 or ans_dec == -1:
			# 		fail = True

			# Check all numbers are integers or easy decimals
			for i in range(0,len(sequence)):
				dec_piece = sequence[i] - int(sequence[i])
				if abs(dec_piece) != 0:
					if abs(dec_piece) != 0.5:
						fail = True
						continue

			# Check numbers are not too large
			for i in range(0,len(sequence)):
				if abs(sequence[i]) > 10000: 
					fail = True
					continue

			fake_ans = GenerateFakeAnswers(sequence)

			# Massage answers so that they have the same significant figures
			# First get number of decimal places of the correct answer
			ans_dec = getDec(sequence[-1])

			# If 'e' is found in the number as a string, assume the number is too high or low to be used and fail the operation
			if ans_dec == -1:
				fail = True
				continue
			else:
				for i in range(0,len(fake_ans)):
					fake_ans[i] = changeDec(fake_ans[i],ans_dec)
					if False in fake_ans:
						fail = True
						continue

			# Check answer is unique amongst fakes
			for i in range(0,len(fake_ans)):
				if fake_ans[i] == sequence[-1]:
					fake_ans[i] = fake_ans[i] + 1

			# If answers are close, increment of decrement them until all are unique.
			for i in range(0,len(fake_ans)):
				for j in range(0,len(fake_ans)):
					if fake_ans[i] == fake_ans[j] and i!=j:
						fake_ans[j] = fake_ans[j] + 1

			diff_array.append(int(diffscore))

		# Try and predict difficulty
		diffscore = CalcDifficulty(diff_array,sequence)

		fake_ans.append(sequence[-1])
		fake_ans.sort()

		sequence_str = ""

		for i in range(0,len(sequence)):
			sequence_str = sequence_str + str(sequence[i]) + "_" 

		if sequence_str in gen_history:
			added = False
			print("Generated sequence before")
			continue
		else:
			added = True
			writeToCSV(itt,sequence,fake_ans,path)

	#print("Have a "+str(layers)+" layered problem")
	#print("Sequence = "+str(sequence[:-1]))	
	#print("Possible Ans = "+str(fake_ans))
	#print("Correct Ans = "+str(sequence[-1]))

	return sequence

############################################################################################
# Generate Problem Set
# Our main function to take our starting parameters to request create a new sequence and create a new folder to store the output. 
############################################################################################
def GenProblemSet(num_problems,num_layers,location):
	currtime = datetime.datetime.now()
	currtime_str = currtime.strftime("%d-%m-%Y.%H.%M.%S")
	new_fold = "Num_Set_"+currtime_str+"/"

	path = location+new_fold

	try:
		os.mkdir(path)
	except OSError:  
		print ("Creation of the directory %s failed" % path)
	else:  
		print ("Successfully created the directory %s " % path)
		for i in range(0,num_problems):
			GenerateSequenceSet(i+1,num_layers,path)


############################################################################################
# Generate Problem Set
# Our main function call
############################################################################################
GenProblemSet(num_problems,num_layers,location)


