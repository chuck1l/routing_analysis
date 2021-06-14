from flr_query import flr_query
from config import database_connect
from important_features import feature_importance

'''
states = ['AL', 'AR', 'CA', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IL', 'IN',
               'KS', 'KY', 'LA', 'MA', 'ME', 'MI', 'MO', 'MS', 'NC', 'NE', 'NJ',
               'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'SC', 'TX', 'UT', 'VA',
               'WA', 'WI', 'WV']
'''

states = ['CA', 'CT', 'DE', 'FL', 'GA', 'ID', 'IN', 'KS', 'KY', 'LA', 'ME', 'MI',
          'MO', 'OH', 'OK', 'PA', 'TX', 'VA']

#states = ['FL', 'CA', 'DE']

conn = database_connect()

for state in states:
    # Perform all tasks for each market
    print(f'Starting analysis for: {state}.')
    flr_query(state, conn)
    #feature_importance(df, state)
    #print(f'{state} file is complete.')


