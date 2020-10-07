bp_gpu = lsub -g 1 -m 22 --autoname --cmd

#General functions
a1 = $(word 1,$(subst _, , $(1)))
a2 = $(word 2,$(subst _, , $(1)))
a3 = $(word 3,$(subst _, , $(1)))
a4 = $(word 4,$(subst _, , $(1)))
a5 = $(word 5,$(subst _, , $(1)))
a6 = $(word 6,$(subst _, , $(1)))

results/test_%: main.py
	$(bp_gpu) python $< $@ --trainset test --S $* --cycle 150 --noise_epochs 120 --sample_epochs 130 --M 8
results/cifar10h_%: main.py
	$(bp_gpu) python $< $@ --trainset cifar10h --S $* --cycle 150 --noise_epochs 120 --sample_epochs 130 --M 8
results/train_%: main.py
	$(bp_gpu) python $< $@ --trainset train --S $*

trainset_list = cifar10h_ test_ 
S_list = 1000 300 100 30 10 3 1

path_trainset = $(addprefix results/,$(trainset_list))
path_trainset_S = $(foreach pre,$(path_trainset),$(addprefix $(pre),$(S_list))) 
cifar10_test: $(path_trainset_S)

cifar10_train: $(addprefix results/train_,$(S_list))