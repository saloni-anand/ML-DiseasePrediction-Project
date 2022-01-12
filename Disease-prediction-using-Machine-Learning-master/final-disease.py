from tkinter import *
import numpy as np
import pandas as pd
import sklearn

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze','fatigue','cough','lethargy','high_fever','nausea','joint_pain','pain_behind_the_eyes','breathlessness']

disease=['Corona Virus','Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
'Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# Training DATA df -------------------------------------------------------------------------------------
df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Corona Virus':0,'Fungal infection':1,'Allergy':2,'GERD':3,'Chronic cholestasis':4,'Drug Reaction':5,
'Peptic ulcer diseae':6,'AIDS':7,'Diabetes ':8,'Gastroenteritis':9,'Bronchial Asthma':10,'Hypertension ':11,
'Migraine':12,'Cervical spondylosis':13,
'Paralysis (brain hemorrhage)':14,'Jaundice':15,'Malaria':16,'Chicken pox':17,'Dengue':18,'Typhoid':19,'hepatitis A':20,
'Hepatitis B':21,'Hepatitis C':22,'Hepatitis D':23,'Hepatitis E':24,'Alcoholic hepatitis':25,'Tuberculosis':26,
'Common Cold':27,'Pneumonia':28,'Dimorphic hemmorhoids(piles)':29,'Heart attack':30,'Varicose veins':31,'Hypothyroidism':32,
'Hyperthyroidism':33,'Hypoglycemia':34,'Osteoarthristis':35,'Arthritis':36,
'(vertigo) Paroymsal  Positional Vertigo':37,'Acne':38,'Urinary tract infection':39,'Psoriasis':40,
'Impetigo':41}},inplace=True) #overwrites existing data without creating copy

print(df.head())

X= df[l1] #training samples

y = df[["prognosis"]] #class labels for the training samples
np.ravel(y) #returns 1d arary
print(y)

# Testing DATA tr --------------------------------------------------------------------------------
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Corona Virus':0,'Fungal infection':1,'Allergy':2,'GERD':3,'Chronic cholestasis':4,'Drug Reaction':5,
'Peptic ulcer diseae':6,'AIDS':7,'Diabetes ':8,'Gastroenteritis':9,'Bronchial Asthma':10,'Hypertension ':11,
'Migraine':12,'Cervical spondylosis':13,
'Paralysis (brain hemorrhage)':14,'Jaundice':15,'Malaria':16,'Chicken pox':17,'Dengue':18,'Typhoid':19,'hepatitis A':20,
'Hepatitis B':21,'Hepatitis C':22,'Hepatitis D':23,'Hepatitis E':24,'Alcoholic hepatitis':25,'Tuberculosis':26,
'Common Cold':27,'Pneumonia':28,'Dimorphic hemmorhoids(piles)':29,'Heart attack':30,'Varicose veins':31,'Hypothyroidism':32,
'Hyperthyroidism':33,'Hypoglycemia':34,'Osteoarthristis':35,'Arthritis':36,
'(vertigo) Paroymsal  Positional Vertigo':37,'Acne':38,'Urinary tract infection':39,'Psoriasis':40,
'Impetigo':41}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test) # returns the array just in 1D, known as flattening of array
# ------------------------------------------------------------------------------------------------------

def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False)) #return the number of correctly classified samples
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        #print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest) #given a trained model, give prediction for new input data
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")


def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y)) #for training the model

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred = clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")


def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")

#knn trial
def kNearestNeighbors():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn import linear_model, preprocessing
    knn = KNeighborsClassifier()
    knn = knn.fit(X, np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred = knn.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    for k in range(0, len(l1)):
        for z in psymptoms:
            if (z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = knn.predict(inputtest)
    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if (predicted == a):
            h = 'yes'
            break

    if (h == 'yes'):
        t5.delete("1.0", END)
        t5.insert(END, disease[a])
    else:
        t5.delete("1.0", END)
        t5.insert(END, "Not Found")


# kmeans
def kMeans():
    from sklearn.preprocessing import scale
    from sklearn.datasets import load_digits
    from sklearn.cluster import KMeans
    from sklearn import metrics
    kmc = KMeans(n_clusters=42, init="random", n_init=10)
    kmc = kmc.fit(X, np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred = kmc.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    for k in range(0, len(l1)):
        for z in psymptoms:
            if (z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = kmc.predict(inputtest)
    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if (predicted == a):
            h = 'yes'
            break

    if (h == 'yes'):
        t4.delete("1.0", END)
        t4.insert(END, disease[a])
    else:
        t4.delete("1.0", END)
        t4.insert(END, "Not Found")

#svm
def sVm():
    from sklearn import svm
    from sklearn import metrics
    from sklearn.neighbors import KNeighborsClassifier
    sv = svm.SVC(kernel="linear", C=2)
    sv = sv.fit(X, np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred = sv.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred, normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]
    for k in range(0, len(l1)):
        for z in psymptoms:
            if (z == l1[k]):
                l2[k] = 1

    inputtest = [l2]
    predict = sv.predict(inputtest)
    predicted = predict[0]

    h = 'no'
    for a in range(0, len(disease)):
        if (predicted == a):
            h = 'yes'
            break

    if (h == 'yes'):
        t6.delete("1.0", END)
        t6.insert(END, disease[a])
    else:
        t6.delete("1.0", END)
        t6.insert(END, "Not Found")


# frequency trial------------------------------------------------------------------------------
def final_predict():
    #    print(t1.get('1.0', END), t2.get('1.0', END), t3.get('1.0', END), t4.get('1.0', END), t5.get('1.0', END), t6.get('1.0', END))
#    arr = [t1.get('1.0', END), t2.get('1.0', END), t3.get('1.0', END), t4.get('1.0', END), t5.get('1.0', END), t6.get('1.0', END)]

    li = list()
    li.append(t1.get('1.0', END))
    li.append(t2.get('1.0', END))
    li.append(t3.get('1.0', END))
    li.append(t4.get('1.0', END))
    li.append(t5.get('1.0', END))
    li.append(t6.get('1.0', END))
    #print(li)
    res = max(set(li), key=li.count)
    #print("Element with highest frequency:\n", res)
    t8.delete("1.0", END)
    t8.insert(END, res)

    print(NameEn.get())
    print(t8.get('1.0', END))

# gui_design------------------------------------------------------------------------------------

root = Tk()
root.configure(background='black')

# entry variables
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Name = StringVar()

# Heading
w2 = Label(root, justify=CENTER, text="Disease Prediction System", fg="white", bg="black")
w2.config(font=("Elephant", 30))
w2.grid(row=1, column=1, columnspan=3, padx=100)
w2 = Label(root, justify=CENTER, text="By Saloni Anand and Kshitij Patne", fg="white", bg="blue")
w2.config(font=("Aharoni", 20))
w2.grid(row=2, column=1, columnspan=3, padx=50)

# labels
NameLb = Label(root, text="Name of the Patient", fg="white", bg="blue")
NameLb.grid(row=6, column=0, pady=15, sticky=W)
# symptoms
S1Lb = Label(root, text="Symptom 1", fg="white", bg="blue")
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="white", bg="blue")
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="white", bg="blue")
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="white", bg="blue")
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="white", bg="blue")
S5Lb.grid(row=11, column=0, pady=10, sticky=W)

# algorithms
ranfLb = Label(root, text="RandomForest", fg="white", bg="red")
ranfLb.grid(row=15, column=0, pady=10, sticky=W)

destreeLb = Label(root, text="DecisionTree", fg="white", bg="red")
destreeLb.grid(row=17, column=0, pady=10, sticky=W)

nbLb = Label(root, text="NaiveBayes", fg="white", bg="red")
nbLb.grid(row=19, column=0, pady=10, sticky=W)

knLb = Label(root, text="KNN", fg="white", bg="red")
knLb.grid(row=23, column=0, pady=10, sticky=W)

kmLb = Label(root, text="KMeans", fg="white", bg="red")
kmLb.grid(row=21, column=0, pady=10, sticky=W)

svecLb = Label(root, text="SVM", fg="white", bg="red")
svecLb.grid(row=25, column=0, pady=10, sticky=W)

predLb = Label(root, text="You Have", fg="white", bg="red")
predLb.grid(row= 27, column=0, pady=10, sticky=W)

# entry_boxes
OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=2)

S1En = OptionMenu(root, Symptom1,*OPTIONS)
S1En.grid(row=7, column=2)

S2En = OptionMenu(root, Symptom2,*OPTIONS)
S2En.grid(row=8, column=2)

S3En = OptionMenu(root, Symptom3,*OPTIONS)
S3En.grid(row=9, column=2)

S4En = OptionMenu(root, Symptom4,*OPTIONS)
S4En.grid(row=10, column=2)

S5En = OptionMenu(root, Symptom5,*OPTIONS)
S5En.grid(row=11, column=2)

# buttons
rnf = Button(root, text="Randomforest", command=randomforest,bg="green",fg="yellow")
rnf.grid(row=8, column=4,padx=10)

dst = Button(root, text="DecisionTree", command=DecisionTree,bg="green",fg="yellow")
dst.grid(row=9, column=4,padx=10)

nb = Button(root, text="NaiveBayes", command=NaiveBayes,bg="green",fg="yellow")
nb.grid(row=10, column=4,padx=10)

kn = Button(root, text="KNN", command=kNearestNeighbors,bg="green",fg="yellow")
kn.grid(row=12, column=4,padx=10)

km = Button(root, text="KMC", command=kMeans,bg="green",fg="yellow")
km.grid(row=11, column=4,padx=10)

svc = Button(root, text="SVM", command=sVm,bg="green",fg="yellow")
svc.grid(row=15, column=4,padx=10)

but = Button(root, text="Prediction", command=final_predict, bg="green", fg="yellow")
but.grid(row=17, column=4,padx=10)

# textfileds
t1 = Text(root, height=1, width=40,bg="white",fg="red")
t1.grid(row=15, column=2 , padx=10)

t2 = Text(root, height=1, width=40,bg="white",fg="red")
t2.grid(row=17, column=2, padx=10)

t3 = Text(root, height=1, width=40,bg="white",fg="red")
t3.grid(row=19, column=2 , padx=10)

t4 = Text(root, height=1, width=40,bg="white",fg="red")
t4.grid(row=21, column=2 , padx=10)

t5 = Text(root, height=1, width=40,bg="white",fg="red")
t5.grid(row=23, column=2 , padx=10)

t6 = Text(root, height=1, width=40,bg="white",fg="red")
t6.grid(row=25, column=2 , padx=10)

t8 = Text(root, height=1, width=40, bg="white",fg="red")
t8.grid(row=27, column=2, padx=10)

root.mainloop()
