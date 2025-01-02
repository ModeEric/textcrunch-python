from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")

text = """The Internet is a global network of interconnected computers that has revolutionized 
        communication and information sharing. It enables users to access a vast array of resources, 
        communicate instantly across great distances, and participate in various online activities. 
        The World Wide Web, often simply called "the Web," is a major part of the Internet that 
        allows users to navigate through hyperlinked documents."""


inputs = tokenizer.encode(text,return_tensors="pt")
for i in range(len(inputs)):
    outputs = model.generate(inputs[:i])
    print(outputs)