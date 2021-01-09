bp_gpu = lsub -g 1 -m 22 -t 12 --autoname --cmd

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
	python $< $@ --trainset train --S $* --net alexnet

trainset_list = cifar10h_ test_ 
S_list = 1000 300 100 30 10 3 1

path_trainset = $(addprefix results/,$(trainset_list))
path_trainset_S = $(foreach pre,$(path_trainset),$(addprefix $(pre),$(S_list))) 
cifar10_test: $(path_trainset_S)

cifar10_train: $(addprefix results/train_,$(S_list))

results/cifar100_%: main.py
	$(bp_gpu) python $< $@ --trainset train --PCIFAR100 $(call a1,$*) --S $(call a2,$*)
c100_list = 0.0_ 0.05_ 0.1_ 0.2_ 0.5_ 0.8_
path_c100 = $(addprefix results/cifar100_,$(c100_list))
path_c100_S = $(foreach pre,$(path_c100),$(addprefix $(pre),$(S_list))) 
cifar100: $(path_c100_S)

results/noise_%: main.py
	$(bp_gpu) python $< $@ --trainset train --Pnoise $(call a1,$*) --S $(call a2,$*)
npath_c100 = $(addprefix results/noise_,$(c100_list))
npath_c100_S = $(foreach pre,$(npath_c100),$(addprefix $(pre),$(S_list))) 
noise: $(npath_c100_S)


results/map_%: main.py
	$(bp_gpu) python $< $@ --trainset train --S $* --net alexnet --type map
path_map_S = $(addprefix results/map_,$(S_list))
map: $(path_map_S)

results/sgd_res_%: sgd.py
	$(bp_gpu) python $< $@ --S $*
path_sgd_S = $(addprefix results/sgd_res_,$(S_list))
sgd: $(path_sgd_S)

#results/train10000_%: sgd.py
#	$(bp_gpu) python $< $@ --S $* 
#path_sgd_S = $(addprefix results/sgd_res_,$(S_list))
#sgd: $(path_sgd_S)


S_gen_list = 1 2 3 4 5 6 7
results/toy_%: toy.py
	$(bp_gpu) python $< $@ --S $* 
path_toy_S = $(addprefix results/toy_,$(S_gen_list))
toy: $(path_toy_S)
	

results/toy_ssl/%: toy_ssl.py
	$(bp_gpu) python $< $@ --S $*
path_toy_ssl_S = $(addprefix results/toy_ssl/,$(S_gen_list))
toy_ssl: $(path_toy_ssl_S)
	
