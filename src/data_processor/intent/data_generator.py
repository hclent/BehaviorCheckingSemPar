"""
Manually written NL-SQL templates for an intent and a given database schema.
"""

import json
import sys


PRODUCT_PLACEHOLDER = '__product_name0__'

products = [
    'air pillow',
    'backpack',
    'basket',
    'bicyle',
    'blanket',
    'bug spray',
    'candle',
    'electric bicyle',
    'flashlight',
    'hiking boots',
    'hiking shoes',
    'pillow',
    'scooter',
    'sleeping bag',
    'tent',
    'water bottle'
]


# --- customers_and_products_contacts --- #
product_price_query = {
    'sql': "select Products.product_price from Products where Products.product_name = \"__product_name0__\"",
    'questions': [
        'What is the price of a __product_name0__?',
        'What is the price of the __product_name0__?',
        'What is the cost of a __product_name0__?',
        'What is the cost of the __product_name0__?',
        'Show me the price of a __product_name0__?',
        'Show me the price of the __product_name0__?',
        'Give me the cost of a __product_name0__?',
        'Give me the cost of the __product_name0__?',
        'How much does a __product_name0__ cost?',
        'How much does the __product_name0__ cost?',
        'How much does our __product_name0__ cost?',
        'How much is a __product_name0__?',
        'How much is the __product_name0__?',
        'What price are we selling __product_name0__ at?'
    ]
}


orders_by_date_query = {
    'sql': "select Customer_Orders.order_id from Customer_Orders "
           "join Order_Items on Customer_Orders.order_id = Order_Items.order_id "
           "join Products on Order_Items.product_id = Products.product_id "
           "where Products.product_name = \"__product_name0__\" order by Customer_Orders.order_date",
    'questions': [
        'show customer orders of __product_name0__ by the order date',
        'show customer orders of __product_name0__ by date',
        'give me the customer orders of __product_name0__ and sort them by the order date',
        'give me the customer orders of __product_name0__ and group them by the order date',
        'orders of __product_name0__ and sort them by the order date',
        'orders of __product_name0__ and sort them by date',
        'orders of __product_name0__ and group them by date',
        '__product_name0__ orders and group by order date',
        '__product_name0__ orders and sort by order date',
        '__product_name0__ orders and group by date',
        '__product_name0__ orders and sort by date',
        '__product_name0__ orders grouped by order date',
        '__product_name0__ orders sorted by order date',
        '__product_name0__ orders by order date',
        '__product_name0__ orders grouped by date',
        '__product_name0__ orders sorted by date',
        '__product_name0__ orders by date',
    ]
}


customers_by_address_query = {
    'sql': "select Customers.customer_name from Customers "
           "join Customer_Orders on Customers.customer_id = Customer_Orders.customer_id "
           "join Order_Items on Customer_Orders.order_id = Order_Items.order_id "
           "join Products on Order_Items.product_id = Products.product_id "
           "where Products.product_name = \"__product_name0__\" order by Customers.customer_address",
    'questions': [
        'Find the customers who ordered __product_name0__ and group them by address',
        'Show the customers who bought __product_name0__ and group them by their locations',
        'show customers who ordered __product_name0__ and group them by location',
        'customers who ordered the __product_name0__ and group them by address',
        'Find the customers who ordered __product_name0__ and group by address',
        'Show the customers who bought __product_name0__ and group by their locations',
        'show customers who ordered __product_name0__ and group by location',
        'customers who ordered the __product_name0__ and group by address'
        '__product_name0__ customers and group by address',
        '__product_name0__ customers and group by location',
        '__product_name0__ customers group by address',
        '__product_name0__ customers group by location',
        '__product_name0__ customers by address',
        '__product_name0__ customers by location'
    ]
}


def gen_synthesized_data():
    db_name = sys.argv[1]
    out_json = sys.argv[2]

    dataset = []
    intents = [
        product_price_query,
        orders_by_date_query,
        customers_by_address_query
    ]

    for intent in intents:
        sql = intent['sql']
        for question in intent['questions']:
            for product_name in products:
                exp = {
                    'db_id': db_name,
                    'query': sql.replace(PRODUCT_PLACEHOLDER, product_name),
                    'question': question.replace(PRODUCT_PLACEHOLDER, product_name)
                }
                dataset.append(exp)

    with open(out_json, 'w') as o_f:
        json.dump(dataset, o_f, indent=4)


if __name__ == '__main__':
    gen_synthesized_data()
