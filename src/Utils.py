from keras.layers import StringLookup

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = StringLookup(vocabulary=vocab, oov_token="")
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

vtt_input_shape = (75, 46, 140, 1)
