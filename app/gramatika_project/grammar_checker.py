from py4j.java_gateway import JavaGateway, GatewayParameters

# Specify the port using GatewayParameters
gateway = JavaGateway(gateway_parameters=GatewayParameters(port=25334))

# Get the GrammarChecker instance from the gateway
grammar_checker = gateway.entry_point

# Call the getGrammarSuggestions method
sentence = "kumain ang bata ng mansanas"
suggestions = grammar_checker.getGrammarSuggestions(sentence)

# Print the suggestions
for suggestion in suggestions:
    print(suggestion)
