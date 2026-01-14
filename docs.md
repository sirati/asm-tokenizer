# Project Structure

## Processing steps
1. Tokenization of individual files into binary representation of tokens and metadata.
2. Merging vocabulary of selection of files, and generate a mapping file for each as well as the definition of the merged vocabulary.
3. Generating files for the dataloader that allow memory mapped IO. This matches functions of same binary compiled with different options

## Important Invariants

The files output of the tokenizer must be alphabetically ordered by function name. Otherwise step 3 will silently produce results not containing most data.
