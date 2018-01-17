label_data = [1,3,4,1,2,4,5,1,3,4,6,9,9,7,1,2,3,4,1,9,9,9,7]
print (label_data)

keep_transitions = 1



if (keep_transitions):
    # Introduce the five different transition types
    transitionDict = {(7, 6): 91, (6, 1): 92, (1, 6): 93, (6, 7): 94}
    prevValue = -1
    value = -1
    preTransitionActivity = -1
    postTransitionActivity = -1
    transitionCounter = 0
    for i in range(len(label_data)):
        if (value == -1):
            value = label_data[i]
        else:
            prevValue = value
            value = label_data[i]
            if (value == 9):
                if (transitionCounter == 0):
                    preTransitionActivity = prevValue
                transitionCounter += 1
            else:
                if (prevValue == 9):
                    postTransitionActivity = value
                    #print("preTransitionActivity ", preTransitionActivity)
                    #print("postTransitionActivity ", postTransitionActivity)
                    if ((preTransitionActivity, postTransitionActivity) in transitionDict):
                        transition_type = transitionDict.get((preTransitionActivity, postTransitionActivity))
                    else:
                        transition_type = 9
                    for j in range(transitionCounter):
                        label_data[i - transitionCounter + j] = transition_type
                    transitionCounter = 0

print (label_data)


#Transitions:
    #Type = 9 is transition
    #1 - sitting_standing 7_6              Type = 91
    #2 - standing_moving (e.g walking) 6_1 Type = 92
    #3 - moving_standing 1_6               Type = 93
    #4 - standing_sitting  6_7             Type = 94


dictionary = {}
dictionary[(1,6)] = 1
print("contains (6,1): ", (6,1) in dictionary)
print("contain (1,6); ", (1,6) in dictionary)
print("Dict: ", dictionary)

#if (preTransitionActivity, postTransitionActivity) in transition_counts:
#    transition_counts[(preTransitionActivity, postTransitionActivity)] += 1
#else:
#   transition_counts[(preTransitionActivity, postTransitionActivity)] = 1



#else:
#if ((preTransitionActivity, postTransitionActivity) in transition_counts):
#   transition_counts[(preTransitionActivity, postTransitionActivity)] += 1
#else:
#    transition_counts[(preTransitionActivity, postTransitionActivity)] = 1



