Onnx : MultiOutUGen {
	var <>id, <>desc;
	*ar { |input_array, num_outputs, id, bypass=0, sample_rate=(-1)|
        ^this.multiNewList(['audio', num_outputs, id, bypass, sample_rate] ++ input_array.asArray)
	}

	*kr { |input_array, num_outputs, id, bypass=0, sample_rate=(-1)|
        ^this.multiNewList(['control', num_outputs, id, bypass, sample_rate] ++ input_array.asArray)
	}

	init { arg argNumOutChannels, argID ... theInputs;
		[argNumOutChannels, argID].postln;
		this.id = argID;
		inputs = theInputs;
		^this.initOutputs(argNumOutChannels, rate);
	}

	*loadModel {|synth, id, path|
		//get the index from SynthDescLib
		var defName = synth.defName.asSymbol;
		var synthIndex = SynthDescLib.global[defName];
		
		if (synthIndex != nil) {
			synthIndex=synthIndex.metadata()[defName][id.asSymbol]['index'];
		}{
			SynthDescLib.read(SynthDef.synthDefDir+/+defName.asString++".scsyndef");
			synthIndex = SynthDescLib.global[defName].metadata()[defName][id.asSymbol]['index'];
		};

		synthIndex.postln;
		if (synthIndex == nil){
			"SynthDef has no metadata.\n".error;
		};

		synthIndex.do{|index|
			[synth.nodeID, index].postln;
			synth.server.sendMsg('/u_cmd', synth.nodeID, index, 'load_model', path);
		}
	}

	checkInputs {
		/* TODO */
		^this.checkValidInputs;
	}

	optimizeGraph {
		// This is called once per UGen during SynthDef construction!
		var metadata;
		// For older SC versions, where metadata might be 'nil'
		this.synthDef.metadata ?? { this.synthDef.metadata = () };
		
		
		metadata = this.synthDef.metadata[this.synthDef.name];
		if (metadata == nil) {
			// Add RTNeural metadata entry if needed:
			metadata = ();
			this.synthDef.metadata[this.synthDef.name] = metadata;
			this.desc = ();
			this.desc[\index] = [this.synthIndex];
		}{
			//if the metadata already existed, that means there are multiple UGens with the same id
			
			this.desc = ();
			if (metadata[this.id.asSymbol]==nil){
				//if the id info is not there, it is an additional id
				this.desc[\index] = [this.synthIndex];
			}{
				//if the symbol is there, it is probably multichannel expansion
				//so we load all the indexes into an array so we can set them all at once
				this.desc[\index] = (metadata[this.id.asSymbol][\index].add(this.synthIndex));
			};
		};

		this.id.notNil.if {
			metadata.put(this.id, this.desc);
		}{
			Error("Each Onnx instance in a Synth must have a unique ID.").throw;
		};
	}
}
