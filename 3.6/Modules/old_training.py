import numpy as np
from global_parameters import MAX_SWAP, MAX_FRAGMENTS, GAMMA, BATCH_SIZE, EPOCHS, TIMES, FEATURES
from rewards import get_init_dist, modify_fragment, bunch_eval
import logging
import pickle as pkl
import winsound
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second


scores = 1. / TIMES
n_actions = MAX_FRAGMENTS * MAX_SWAP + 1

losses = []

# Train actor and critic networks
def train(X, actor, critic, decodings, out_dir=None):

    hist = []
# =============================================================================
#    dist = get_init_dist(X, decodings)
#    np.save('./dist_exp.npy',dist)
#    logging.info("Printing initial distribution")
#    print(dist)
# =============================================================================
    #logging.info("dist.npy has been loaded")
#    dist = np.load('./dist.npy')
#    m = X.shape[1]


    # For every epoch
    for e in range(EPOCHS):

        # Select random starting "lead" molecules; number of molecules = BATCH_SIZE; they are sampled from the lead molecules set
        print("Epoch {} starting".format(e))
        rand_n = np.random.randint(0,X.shape[0],BATCH_SIZE)
        #Taking 512 random molecules from the lead set
        batch_mol = X[rand_n].copy()
        #1*512(maybe 512x1) zero vector
        r_tot = np.zeros(BATCH_SIZE)
        org_mols = batch_mol.copy() #saving a copy of the original molecules
        stopped = np.zeros(BATCH_SIZE) != 0


        # For all modification steps
        for t in range(TIMES):

            #for each mol, a no. between 0-1 indicating the time-step
            tm = (np.ones((BATCH_SIZE,1)) * t) / TIMES
            #predictions for all the 512 molecules: 512*n_actions
            probs = actor.predict([batch_mol, tm])
            actions = np.zeros((BATCH_SIZE))
            #Random numbers between 0 and 1
            rand_select = np.random.rand(BATCH_SIZE)
            old_batch = batch_mol.copy()
            rewards = np.zeros((BATCH_SIZE,1))


            # Find probabilities for modification actions
            for i in range(BATCH_SIZE):#for every molecules in the batch

                a = 0
                while True:
                    rand_select[i] -= probs[i,a]
                    if rand_select[i] < 0 or a + 1 == n_actions:
                        break
                    a += 1
                #choosing a random action
                actions[i] = a

            # Initial critic value(predicts the optimal state value)
            Vs = critic.predict([batch_mol,tm])


            # Select actions
            for i in range(BATCH_SIZE):

                a = int(actions[i])
                if stopped[i] or a == n_actions - 1:
                    stopped[i] = True
                    if t == 0:
                        rewards[i] += -1. #Why and why -1?

                    continue


                #Converting the n_actions*1 position to the actual position    
                a = int(a // MAX_SWAP)#Integer Division, to get the location of the fragment

                s = a % MAX_SWAP# it is the location where the swap happens in that fragment
                if batch_mol[i,a,0] == 1:#Checking whether the fragment is non-empty?
                    
                    #In ith molecule, in its ath fragment, the sth position is flipped

                    batch_mol[i,a] = modify_fragment(batch_mol[i,a], s)#changes 0 to 1 and 1 to 0
                else:
                    rewards[i] -= 0.1   #why 0.1? CHANGED THIS TO 10

            # If final round
            if t + 1 == TIMES:
                frs = []
                print ("Final round of epoch {}".format(e))
                modified_mols = []
                org_mole = []
                for i in range(batch_mol.shape[0]):
# =============================================================================
#                     if ((i+1)*100/512)%25==0:
#                         print("Evaluation Comepletion {}%".format(((i+1)*100/512)))
# =============================================================================
                    # If molecule was modified
                    if not np.all(org_mols[i] == batch_mol[i]):
                        modified_mols.append([batch_mol[i],i])
                        org_mole.append([org_mols[i],i])
                        # fr = evaluate_mol(batch_mol[i], e, decodings)
                        # frs.append(fr)
                        # rewards[i] += np.sum(fr * dist)

                        #if all(fr):
                            #rewards[i] *= 2
                    else:
                        frs.append([False,-1])
                        rewards[i] -= 1
                
                #Storing all modified molecules
                print("Length of modified molecules",len(modified_mols))
                molecules = []
                for i in range(len(modified_mols)):
                    molecules.append(modified_mols[i][0])
                evaluation = bunch_eval(molecules,e,decodings)
                if len(molecules)!=0:
                # print(len(molecules),molecules[0])
                    molecules = []
                    for i in range(len(modified_mols)):
                        molecules.append(org_mole[i][0])    
                    org_mole_evaluation = bunch_eval(molecules,e,decodings)
                    for i in range(len(evaluation)):
                        if (evaluation[i,0]) == True:
                            evaluation[i,1] = evaluation[i,1]-org_mole_evaluation[i,1]
                else:
                    evaluation = np.zeros(shape=(len(batch_mol),2),dtype='float64')
                    evaluation[0] = np.full(shape=(len(batch_mol),2),fill_value=0)
                    evaluation[1] = np.full(shape=(len(batch_mol),2),fill_value=-10)
                with open('./evaluation.pkl','wb') as f:
                    pkl.dump(evaluation,f)

                print("Shape of returned evaluations:{}".format(evaluation.shape))
                #Updating Rewards
                for i in range(len(modified_mols)):
                    fr = float(evaluation[i,1])
                    frs.append(fr)
                    rewards[modified_mols[i][1]] += fr

                print("Shape of rewards",rewards.shape)
                # Update distribution of rewards
              #  dist = 0.5 * dist + 0.5 * (1.0/FEATURES * BATCH_SIZE / (1.0 + np.sum(frs,0)))


            # Calculate TD-error
            target = rewards + GAMMA * critic.predict([batch_mol, tm+1.0/TIMES])
            td_error = target - Vs

            # Minimize TD-error
            critic.fit([old_batch,tm], target, verbose=0)
            target_actor = np.zeros_like(probs)


            for i in range(BATCH_SIZE):

                a = int(actions[i])
                loss = -np.log(probs[i,a]) * td_error[i] #looks same as models.maximisation()
                target_actor[i,a] = td_error[i]#This is not the 'target' of the actor but rather a qty used to calculate error fn

            # Maximize expected reward.
            actor.fit([old_batch,tm], target_actor, verbose=0)
           
            r_tot += rewards[:,0]
        #np.save("rewards.npy",rewards)
        #np.save("./Losses/Loss in epoch {}.npy".format(e),loss)
        losses.append(loss)
        np.save("History/in-{}.npy".format(e), org_mols)
        np.save("History/out-{}.npy".format(e), batch_mol)
        np.save("History/score-{}.npy".format(e), np.asarray(frs))
        if(e%50==0):
            np.save("./Losses/Loss.npy",losses)
            np.save("History/history.npy",hist)
            actor.save('./saved_models/generation')
        hist.append([np.mean(r_tot)])#CHANGED FROM 4 to 2
        print ("Epoch {1} \t Mean score: {0:.3}".format(
            np.mean(r_tot), e,#Removed a round for loop
        ))
        

    return hist
