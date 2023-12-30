object(Word) --> 
	my_member(
		Word, 
		[шоколад, деньги, слава, марафон, хайп]
	), 
	[Word]. 

agent(Word) --> 
	my_member(
		Word, 
		[даша, дима, маша, сережа]
	), 
	[Word].

verb(Word, Infinitive) --> 	
	my_member(
		verb_term(Word, Infinitive), 
		[verb_term(делает, делать), verb_term(делают, делать), 
		 verb_term(едет, ехать), verb_term(едут, ехать), 
		 verb_term(хочет, хотеть), verb_term(хотят, хотеть), 
		 verb_term(любит, любить), verb_term(любят, любить),
		 verb_term(бежит, бежать), verb_term(бегут, бежать)]
	), 
	[Infinitive]. 
						 
question(Word, Meaning) --> 
	my_member(
		question_term(Word, Meaning), 
		[question_term(кто, agent), question_term(что, object),
		 question_term(где, loc), question_term(как, method),
		 question_term(когда, time)]
	), 
	[Meaning]. 
						
anim(Word, animate) --> 	
	my_member(
		Word, 
		[даша, дима, маша, сережа]
	), 
	[animate]. 

anim(Word, inanimate) --> 	
	my_member(
		Word, 
		[шоколад, деньги, слава, марафон, хайп]
	), 
	[inanimate]. 
