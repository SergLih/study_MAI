#coding: utf-8
import re

personID, personFirstName, personLastName, personGender = None, None, None, None
persons = dict()

inFile = open("data.ged", "r", encoding='utf8')

print('input file processing...', end="")
for string in inFile:
    reSearchRes = re.search(r"^0\s+@I(\d+)@\s+INDI$", string)  # получаем id человека
    if reSearchRes is not None:
        personID = reSearchRes.group(1)
        continue

    reSearchRes = re.search(r"^2\s+GIVN\s+(.+)$", string) # имя
    if reSearchRes is not None:
        personFirstName = reSearchRes.group(1)
        continue
    
    reSearchRes = re.search(r"^2\s+SURN\s+(.+)$", string) # фамилия
    if reSearchRes is not None:
        personLastName = reSearchRes.group(1)
        continue
    else:  # если нет девичьей фамилии, ищем после свадьбы
        reSearchRes = re.search(r"^2\s+_MARNM\s+(.+)$", string) # фамилия после свадьбы
        if reSearchRes is not None:
            personLastName = reSearchRes.group(1)
            continue

    reSearchRes = re.search(r"^1\s+SEX\s+([F|M])$", string)  # пол
    if reSearchRes is not None:
        personGender = reSearchRes.group(1)
        persons[personID] = [personFirstName, personLastName, personGender, []]
        personID, personFirstName, personLastName, personGender = None, None, None, None
        continue

        
    # раздел где указана информация о детях   
     
    reSearchRes = re.search(r"0\s+@F\d+@\s+FAM$", string)  # новая семья
    if reSearchRes is not None:
        fatherID, motherID = None, None
        continue
        
    reSearchRes = re.search(r"1\s+HUSB\s+@I(\d+)@$", string)  # получение id мужа
    if reSearchRes is not None:
        fatherID = reSearchRes.group(1)
        continue

    reSearchRes = re.search(r"1\s+WIFE\s+@I(\d+)@$", string)  # получение id жены
    if reSearchRes is not None:
        motherID = reSearchRes.group(1)
        continue

    reSearchRes = re.search(r"1\s+CHIL\s+@I(\d+)@$", string)  # запись ребенка
    if reSearchRes is not None:
        childID = reSearchRes.group(1)
        if fatherID:
            persons[fatherID][3].append(childID)
            print('append father(', fatherID, childID, ')')
        if motherID:
            persons[motherID][3].append(childID)
            print('append mother(', motherID, childID, ')')
        print()
        continue

inFile.close()

lines = []
for personID, personData in persons.items():
    lines.append('person({}, "{} {}").'.format(personID, *personData[:2]))
    
for personID, personData in persons.items():
    if personData[3]:  # если непустой список детей, пройти по нему
        for childID in personData[3]:  
            lines.append("{}({}, {}).".format(( 'father' if personData[2]=='M' else 'mother'), personID, childID))  
    else:
        lines.append("{}({}, {}).".format(( 'father' if personData[2]=='M' else 'mother'), personID, 0))  

lines.sort()
print('\nsaving results...', end='')
outFile = open("database.pl", "w")
outFile.write('\n'.join(lines))
outFile.close()
print('done.', end='')
