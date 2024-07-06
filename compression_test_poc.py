import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import zlib
import struct
from bitarray import bitarray
import matplotlib.pyplot as plt
from collections import Counter

def get_sample_texts():
    return [
        """The Internet is a global network of interconnected computers that has revolutionized 
        communication and information sharing. It enables users to access a vast array of resources, 
        communicate instantly across great distances, and participate in various online activities. 
        The World Wide Web, often simply called "the Web," is a major part of the Internet that 
        allows users to navigate through hyperlinked documents.""",

        """Artificial Intelligence (AI) is a branch of computer science focused on creating intelligent 
        machines that can perform tasks typically requiring human intelligence. These tasks include 
        visual perception, speech recognition, decision-making, and language translation. Machine 
        learning, a subset of AI, involves algorithms that can learn from and make predictions or 
        decisions based on data.""",
    ]

def load_gpt2():
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return model, tokenizer

def encode_bit_level(value):
    if value < 4:  # 2-bit encoding for 0-3
        return bitarray(f'{value:02b}')
    elif value < 16:  # 4-bit encoding for 4-15
        return bitarray('10') + bitarray(f'{value:04b}')
    else:  # 8-bit or more encoding for 16+
        binary = f'{value:b}'
        length_bits = bitarray('1' * (len(binary) - 1) + '0')
        return bitarray('11') + length_bits + bitarray(binary)

def decode_bit_level(bits):
    if bits[0] == 0:  # 2-bit encoding
        return int(bits[:2].to01(), 2)
    elif bits[1] == 0:  # 4-bit encoding
        return int(bits[2:6].to01(), 2)
    else:  # 8-bit or more encoding
        length = bits[2:].index(0) + 1
        return int(bits[length+2:length+2+length].to01(), 2)

# The rest of the code remains the same
def llm_compress_bit_level(text, model, tokenizer, window_size=128):
    model.eval()
    encoded = tokenizer.encode(text, add_special_tokens=True)  # This will add the EOS token
    compressed = bitarray()
    ranks = []
    
    for i in range(len(encoded)):
        input_ids = torch.tensor([encoded[max(0, i - window_size):i]])
        if input_ids.numel() == 0:
            compressed.extend(encode_bit_level(encoded[i]))
            ranks.append(encoded[i])
        else:
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
            
            next_token_logits = logits[0, -1, :]
            sorted_indices = torch.argsort(next_token_logits, descending=True)
            
            actual_token = encoded[i]
            rank = (sorted_indices == actual_token).nonzero().item()
            
            compressed.extend(encode_bit_level(rank))
            ranks.append(rank)
    
    # Pad to full byte
    compressed.extend([0] * ((8 - len(compressed) % 8) % 8))
    
    return struct.pack('>I', len(compressed) // 8) + compressed.tobytes(), ranks
def llm_decompress_bit_level(compressed_data, model, tokenizer, window_size=128):
    model.eval()
    
    compressed_length = struct.unpack('>I', compressed_data[:4])[0]
    bits = bitarray()
    bits.frombytes(compressed_data[4:4+compressed_length])
    
    decoded = []
    bit_index = 0
    
    while bit_index < len(bits):
        value = decode_bit_level(bits[bit_index:])
        bit_index += len(encode_bit_level(value))
        
        if len(decoded) == 0:
            next_token = value
        else:
            if len(decoded) < window_size:
                input_ids = torch.tensor([decoded])
            else:
                input_ids = torch.tensor([decoded[-window_size:]])
            
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
            
            next_token_logits = logits[0, -1, :]
            sorted_indices = torch.argsort(next_token_logits, descending=True)
            
            next_token = sorted_indices[value].item()
        
        decoded.append(next_token)
        
        # Early stopping condition
        if tokenizer.decode([next_token]) == tokenizer.eos_token:
            break
    
    return tokenizer.decode(decoded)

def plot_rank_distribution(ranks):
    plt.figure(figsize=(12, 6))
    plt.hist(ranks, bins=50, edgecolor='black')
    plt.title('Distribution of Ranks')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()

    # Calculate and print some statistics
    rank_counter = Counter(ranks)
    total_ranks = len(ranks)
    ranks_0_3 = sum(rank_counter[i] for i in range(4))
    ranks_4_15 = sum(rank_counter[i] for i in range(4, 16))
    ranks_16_plus = total_ranks - ranks_0_3 - ranks_4_15

    print(f"Ranks 0-3: {ranks_0_3} ({ranks_0_3/total_ranks:.2%})")
    print(f"Ranks 4-15: {ranks_4_15} ({ranks_4_15/total_ranks:.2%})")
    print(f"Ranks 16+: {ranks_16_plus} ({ranks_16_plus/total_ranks:.2%})")

def main():
    model, tokenizer = load_gpt2()
    texts = get_sample_texts()

    all_ranks = []

    for i, text in enumerate(texts, 1):
        print(f"\n{'='*50}")
        print(f"Text {i}:")
        print(f"{'='*50}")
        
        print("Original text:")
        print(text)
        print(f"{'='*50}")
        
        original_length = len(text.encode('utf-8'))
        print("Original text length:", original_length, "bytes")

        llm_compressed, ranks = llm_compress_bit_level(text, model, tokenizer)
        all_ranks.extend(ranks)
        print("LLM compressed length:", len(llm_compressed), "bytes")

        zlib_compressed = zlib.compress(text.encode('utf-8'))
        print("zlib compressed length:", len(zlib_compressed), "bytes")

        llm_ratio = len(llm_compressed) / original_length
        zlib_ratio = len(zlib_compressed) / original_length
        print(f"LLM compression ratio: {llm_ratio:.2f}")
        print(f"zlib compression ratio: {zlib_ratio:.2f}")

        decompressed_text = llm_decompress_bit_level(llm_compressed, model, tokenizer)
        print(f"{'='*50}")
        print("Decompressed text:")
        print(decompressed_text)
        print(f"{'='*50}")
        
        is_successful = decompressed_text.strip() == text.strip()
        print("Decompression successful:", is_successful)
        
        if not is_successful:
            print("\nDifferences found:")
            original_lines = text.strip().split('\n')
            decompressed_lines = decompressed_text.strip().split('\n')
            for j, (orig, decomp) in enumerate(zip(original_lines, decompressed_lines)):
                if orig != decomp:
                    print(f"Line {j+1}:")
                    print(f"Original: {orig}")
                    print(f"Decompressed: {decomp}")
                    print()

    # Plot the distribution of ranks
    plot_rank_distribution(all_ranks)

if __name__ == "__main__":
    main()