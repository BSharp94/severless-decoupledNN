
def my_function(event, context):
    message = 'Hello {} {}!'.format(event['first_name'], 
                                    event['last_name'])  
    return { 
        'message' : message
    }