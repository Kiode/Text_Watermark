import argparse
# TODO: add help for the parameters

def get_parser_main_model():
    parser = argparse.ArgumentParser()
    # TODO: basic parameters training related
    
    # for embed 
    parser.add_argument('--language', type=str, default='English', help='text language')
    parser.add_argument('--mode', type=str, choices=['embed', 'fast_detect', 'precise_detect'], default='embed', help='Mode options: embed (default), fast_detect, precise_detect')
    parser.add_argument('--tau_word', type=float, default=0.8, help='word-level similarity thresh')  
    parser.add_argument('--lamda', type=float, default=0.83, help='word-level similarity weight')  
         
    return parser
