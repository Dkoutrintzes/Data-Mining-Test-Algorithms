#pip3 install mlxtend  
#pip3 install xlrd

import sys
import numpy as np 
import pandas as pd 
import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
#######################################################################
#### Def's that we are going to use for creating the apriori rules ####
#######################################################################
def create_basket(Country):
	# Transactions done in a country
	basket = (data[data['Country'] == Country]
					 .groupby(['InvoiceNo', 'Description'])['Quantity']
					 .sum().unstack().reset_index().fillna(0)
					 .set_index('InvoiceNo'))
	basket = encoding(basket)
	return basket

def hot_encode(x):
	# Defining the hot encoding function to make the data suitable
	# for the concerned libraries
	if(x<= 0):
		return 0
	if(x>= 1):
		return 1
def encoding(basket):
	# Enctoding the datasets
	basket_encoded = basket.applymap(hot_encode)
	basket = basket_encoded
	return basket
def built_model(basket,support):
	# Building the model
	frq_items = apriori(basket, min_support=support, use_colnames=True)
	return frq_items

def create_rules(Country,threashold,support):
	# Creating the apriori rules
	basket = create_basket(Country)
	frq_items = built_model(basket,support)
	rules = association_rules(frq_items, metric="lift", min_threshold=threashold)
	rules = rules.sort_values(['lift','confidence'], ascending=[False, False])
	return rules

#######################################################################
############################ PreProcessing ############################
#######################################################################
#np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None) # or 1000
pd.set_option('display.max_rows', None) # or 1000
pd.set_option('display.max_colwidth', -1) # or 199

# Loading the Data 
data = pd.read_excel('Online_Retail.xlsx')

# Exploring the columns of the data
data.columns 
# Exploring the different regions of transactions 
print(data.Country.unique() )

#Cleanning the data
# Stripping extra spaces in the description 
data['Description'] = data['Description'].str.strip() 

# Dropping the rows without any invoice number 
data.dropna(axis = 0, subset =['InvoiceNo'], inplace = True) 
data['InvoiceNo'] = data['InvoiceNo'].astype('str') 

# Dropping all transactions which were done on credit 
data = data[~data['InvoiceNo'].str.contains('C')]
#######################################################################
##### At this point our dataset is ready for the apriori agorithm #####
#######################################################################


# Checking apriori rules for the transactions in Netherlands
rules = create_rules('Netherlands',20,0.035)
print("The Apriori Rules Form Netherlands Are")
print(rules.head(5),'\n')

# Checking apriori rules for the transactions in Norway
rules = create_rules('Norway',0,0.05)
print("The Apriori Rules Form Norway Are")
print(rules.head(5),'\n')

# Checking apriori rules for the transactions in Spain
rules = create_rules('Spain',0,0.05)
print("The Apriori Rules Form Spain Are")
print(rules.head(5),'\n')

# Checking apriori rules for the transactions in Portugal
rules = create_rules('Portugal',5,0.02)
print("The Apriori Rules Form Portugal Are")
print(rules.head(5),'\n')

# Checking apriori rules for the transactions in Poland
rules = create_rules('Poland',5,0.025)
print("The Apriori Rules Form Poland Are")
print(rules.head(5),'\n')

