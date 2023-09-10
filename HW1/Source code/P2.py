import itertools

import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model

data = pd.read_csv(".\Inputs\insurance.csv")


def main():
    # part_a()
    part_b()
    # part_c()
    # part_d()
    # part_e()
    # part_f()
    # part_g()
    # part_h()


def part_a():
    sex = data['sex']

    counted = pd.value_counts(sex)
    print(counted)

    # uncomment for test
    plt.pie(counted, labels=counted.keys())
    plt.show()

def part_b():
    childs = data['children']
    charges = data['charges']
    plt.title("Distribution of childs/charge")
    plt.xlabel("childs")
    plt.ylabel("charges")
    plt.scatter(childs, charges)
    plt.show()

    all_childs_numer = childs.unique()
    all_childs_numer.sort()
    avg_charges = []
    for i in all_childs_numer:
        temp_data = data.loc[data['children'] == i]
        avg_charges.append(temp_data['charges'].mean())

    print(all_childs_numer)

    plt.title("Distribution of childs/charge")
    plt.xlabel("childs")
    plt.ylabel("charges")
    plt.bar(all_childs_numer, numpy.array(avg_charges))

    plt.show()


def part_c():
    regions = data['region'].unique()
    count = 0
    fig, axs = plt.subplots(2, 2, figsize=(20, 15))
    fig.suptitle('Q 2 part c', fontsize=16)
    for reigon in regions:
        tempData = data.loc[data['region'] == reigon]
        all_childs_number = tempData['children'].unique()
        all_childs_number.sort()
        avg_charges = []
        for i in all_childs_number:
            temp_data = tempData[tempData['children'] == i]
            avg_charges.append(temp_data['charges'].mean())

        if count > 1:
            axs[1, count % 2].title.set_text(reigon)
            axs[1, count % 2].bar(all_childs_number, avg_charges)
        else:
            axs[0, count % 2].title.set_text(reigon)
            axs[0, count % 2].bar(all_childs_number, avg_charges)
        count += 1

    plt.show()


def part_d():
    BMI = data['bmi']
    max_bmi = BMI.max()
    min_bmi = BMI.min()
    bins_number = int((max_bmi - min_bmi) / 1.5)
    plt.hist(BMI, fc="#AAAAFF", density=True, bins=bins_number)
    plt.title("distribution of density of BMI")
    plt.xlabel("BMI")
    plt.ylabel("density")
    plt.show()


def part_e():
    AGE = data['age']
    max_bmi = AGE.max()
    min_bmi = AGE.min()
    bins_number = int((max_bmi - min_bmi) / 5)  # setup size of each bin in this case is 5
    plt.hist(AGE, fc="#AAAAFF", density=True, bins=bins_number)
    plt.title("distribution of density of age")
    plt.xlabel("age")
    plt.ylabel("density")
    plt.show()


def part_f():
    data_smokers = data.loc[data['smoker'] == "yes"]
    data_no_smokers = data.loc[data['smoker'] == "no"]

    # print(data_smokers)

    plt.scatter(data_smokers['age'], data_smokers['charges'], label="smoker", alpha=0.4)
    plt.scatter(data_no_smokers['age'], data_no_smokers['charges'], label="non smoker", alpha=0.4)
    plt.legend()
    plt.title("smoking vs non smokings in number of charges")
    plt.ylabel('charges')
    plt.xlabel('age')
    plt.show()

    column_names = ['smoker', 'age', 'charges']

    group_smoker = data_smokers.groupby(pd.cut(data_smokers["age"], np.arange(10, 100, 10))).mean()
    group_non_smoker = data_no_smokers.groupby(pd.cut(data_no_smokers["age"], np.arange(10, 100, 10))).mean()

    print(group_non_smoker)
    bins = np.linspace(19, 89, 8)

    bins2 = np.linspace(21,91,8)

    plt.title("smoking vs non smokings in number of charges")
    plt.ylabel('charges')
    plt.xlabel('age')
    plt.bar(bins,group_smoker['charges'], alpha=0.5, label='smokers' ,width=3)
    plt.bar(bins2,group_non_smoker['charges'], alpha=0.5, label='non-smokers' ,width=3)
    plt.legend()
    plt.show()

    # sns.pairplot(data[column_names])
    # plt.show()


def part_g():
    data_female = data.loc[data['sex'] == "female"]
    data_male = data.loc[data['sex'] == "male"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.suptitle("influence of smoking in male and females")

    ax1.title.set_text("part one")
    ax1.set_ylabel("charges")
    avg_charges_female = data_female.loc[data_female['smoker'] == 'yes']['charges'].mean()
    avg_charges_male = data_male.loc[data_male['smoker'] == 'yes']['charges'].mean()
    ax1.bar(["male smoker", "female smoker"], [avg_charges_male, avg_charges_female])

    ax2.title.set_text("part two")
    avg_charges_male_smoker = data_male.loc[data_male['smoker'] == 'yes']['charges'].mean()
    avg_charges_male_no_smoke = data_male.loc[data_male['smoker'] == 'no']['charges'].mean()
    ax2.bar(["male smoker", "male non smoker"], [avg_charges_male_smoker, avg_charges_male_no_smoke])
    # print(avg_charges_female)
    plt.show()


# in this part we draw diagram in one plot with share Y for comparing
# elements
def part_h():
    regions = data['region'].unique()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    res = []

    for region in regions:
        region_data = data.loc[data['region'] == region]
        no_smokers = region_data.loc[region_data['smoker'] == "yes"]
        mean_charges = no_smokers.loc[no_smokers['children'] >= 3]['charges'].mean()
        res.append(mean_charges)

    print(res)
    ax1.title.set_text("smokers with 3 or more childs")
    ax1.set_ylabel("avg charges")
    ax1.bar(regions, res)

    res = []
    for region in regions:
        region_data = data.loc[data['region'] == region]
        no_smokers = region_data.loc[region_data['smoker'] == "no"]
        mean_charges = no_smokers.loc[no_smokers['children'] >= 4]['charges'].mean()
        res.append(mean_charges)

    ax2.bar(regions, res)
    ax2.title.set_text("no_smokers with 4 or more childs")
    plt.show()


if __name__ == '__main__':
    main()
