# list1=[1,2,3,4,5,3,5]
# list1.append(31)

# tuple1=(1,2,3,4,5)

# dict={"Key1":"Value1","Key2":"Value2"}

# set1={1,2,3,3,3,3,45,65,4,5,2,2,4}

# print(set1)

# str="I'm dharshan... I have B.tech and Diploma degree!!!"
# words=str.split()

# print(words)

# def helloWorld(name):
#     print(name)

# helloWorld("Aditya Chola")

import string

def preprocess1(inputText):
    inputText=inputText.lower()

    for punct in string.punctuation:
        inputText=inputText.replace(punct,"")
    
    return inputText

result=preprocess1("I'm dharshan... I'm studying B.tech degree!!! B.tech is good degree..")
print("Preprocessed input :", result)

counts={}

def wordCount(inputText):
    words=inputText.split()

    for word in words:
        counts[word]=counts.get(word,0) + 1
        
    return counts

print(wordCount(result))

sorted_count=sorted(counts.items(), key=lambda x: x[1], reverse=True)
top3=sorted_count[:3]
print("Top 3 :", top3)