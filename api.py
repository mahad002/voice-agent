import json
import uuid
from datetime import datetime
from pathlib import Path
import os
from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
import openai
import logging
from difflib import get_close_matches

app = Flask(__name__)

# Setup Logger
def make_logger(log_dir="logs", log_name="ovc", console_log=True):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    log_file = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    if console_log:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s,%(name)s,%(levelname)s,%(message)s,%(details)s,%(further)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

logger = make_logger(console_log=True)

class CustomerServiceSystem:
    def __init__(self):
        load_dotenv(dotenv_path=Path('.') / '.env')
        
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.openai_client = openai.OpenAI(api_key=self.OPENAI_API_KEY)

        for folder in ["responses", "reviews", "orders", "meetings"]:
            Path(folder).mkdir(exist_ok=True)

        self.store_info = self.load_json_file('description.json', default={
            "store_name": "Unknown Store",
            "store_description": "No description available.",
            "product_categories": []
        })
        self.products = self.load_json_file('products.json', default=[
            {"name": "Laptop", "quantity": 10}, 
            {"name": "Phone", "quantity": 15}
        ])  # Example with quantity
        self.staff = self.load_json_file('staff.json', default=[])

        self.conversation_states = {}

    def load_json_file(self, filename, default):
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"{filename} not found. Using default values.")
            return default

    def save_products(self):
        with open('products.json', 'w') as f:
            json.dump(self.products, f)

    def normalize_time(self, time_str):
        time_str = time_str.lower().strip().replace('.', '').replace('  ', ' ')
        if 'am' in time_str or 'pm' in time_str:
            time_str = time_str.replace(' am', 'am').replace(' pm', 'pm')
            hours = time_str.replace('am', '').replace('pm', '').strip()
            if ':' not in hours:
                hours = f"{hours}:00"
            return f"{hours} {'AM' if 'am' in time_str else 'PM'}"
        if time_str.isdigit():
            hour = int(time_str)
            return f"{hour}:00 AM" if hour < 12 else f"{hour-12}:00 PM" if hour > 12 else "12:00 PM"
        return time_str.upper()

    def extract_time_from_text(self, text):
        text = text.lower().split()
        for i, word in enumerate(text):
            if any(period in word for period in ['am', 'pm']):
                return self.normalize_time(word)
            if word in ['at', 'for', 'around'] and i + 1 < len(text):
                next_word = text[i + 1]
                if next_word.replace(':', '').replace('.', '').isdigit():
                    return self.normalize_time(next_word if i + 2 >= len(text) or 'am' not in text[i + 2].lower() else f"{next_word} {text[i + 2]}")
        return None

    def get_conversation_state(self, session_id):
        if session_id not in self.conversation_states:
            self.conversation_states[session_id] = {'state': None, 'data': {}, 'history': []}
        return self.conversation_states[session_id]

    def stream_openai_response(self, prompt, session_id):
        state = self.get_conversation_state(session_id)
        state['history'].append({"role": "user", "content": prompt})
        
        response_stream = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful customer service assistant for a store."},
                *state['history']
            ],
            stream=True
        )
        
        def generate():
            full_response = ""
            for chunk in response_stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            state['history'].append({"role": "assistant", "content": full_response})
        
        return generate

    def match_product(self, user_input):
        product_names = [p["name"].lower() for p in self.products]
        matches = get_close_matches(user_input.lower(), product_names, n=1, cutoff=0.6)
        return matches[0] if matches else None

    def handle_query(self, user_input, session_id):
        state = self.get_conversation_state(session_id)
        
        # Meeting Handling
        if "meeting" in user_input.lower() and state['state'] is None:
            state['state'] = 'meeting_requested'
            staff_names = ", ".join([s["name"] for s in self.staff])
            return self.stream_openai_response(f"Available staff members are: {staff_names}. Please specify who you'd like to meet with and your preferred time.", session_id)
        
        if state['state'] == 'meeting_requested':
            if 'selected_staff' in state['data']:
                matched_staff = state['data']['selected_staff']
                extracted_time = self.extract_time_from_text(user_input)
                if extracted_time:
                    for time in matched_staff['availability']:
                        if self.normalize_time(time) == extracted_time:
                            meeting_id = str(uuid.uuid4())
                            meeting_data = {
                                'id': meeting_id,
                                'staff': matched_staff['name'],
                                'time': time,
                                'created_at': datetime.now().isoformat()
                            }
                            with open(Path('meetings') / f'{meeting_id}.json', 'w') as f:
                                json.dump(meeting_data, f)
                            state['state'] = None
                            state['data'] = {}
                            return self.stream_openai_response(f"Great! I've scheduled your meeting with {matched_staff['name']} at {time}.", session_id)
                    available_times = ", ".join(matched_staff['availability'])
                    return self.stream_openai_response(f"I couldn't match that time. Available times for {matched_staff['name']} are: {available_times}. Please specify your preferred time.", session_id)
            else:
                for staff in self.staff:
                    if staff['name'].lower() in user_input.lower():
                        state['data']['selected_staff'] = staff
                        available_times = ", ".join(staff['availability'])
                        return self.stream_openai_response(f"I found {staff['name']}. Their available times are: {available_times}. Please specify your preferred time.", session_id)
                staff_names = ", ".join([s["name"] for s in self.staff])
                return self.stream_openai_response(f"I couldn't find that staff member. Available staff are: {staff_names}. Please try again.", session_id)

        # Order Handling
        if "order" in user_input.lower():
            if state['state'] is None:
                matched_product = self.match_product(user_input)
                if matched_product:
                    state['state'] = 'confirm_product'
                    state['data']['product'] = matched_product
                    return self.stream_openai_response(f"Did you mean {matched_product}? Please say 'yes' or 'no'.", session_id)
                return self.stream_openai_response("We don’t have that item. Could you repeat or try something else?", session_id)
            
            elif state['state'] == 'confirm_product':
                if "yes" in user_input.lower():
                    state['state'] = 'request_quantity'
                    return self.stream_openai_response(f"Great! How many {state['data']['product']}s would you like?", session_id)
                state['state'] = None
                state['data'] = {}
                return self.stream_openai_response("Okay, let’s try again. What would you like to order?", session_id)
            
            elif state['state'] == 'request_quantity':
                try:
                    quantity = int(user_input.strip())
                    product_name = state['data']['product']
                    for product in self.products:
                        if product['name'].lower() == product_name:
                            if product['quantity'] >= quantity:
                                product['quantity'] -= quantity
                                self.save_products()
                                order_id = str(uuid.uuid4())
                                order_data = {
                                    'id': order_id,
                                    'product': product_name,
                                    'quantity': quantity,
                                    'created_at': datetime.now().isoformat()
                                }
                                with open(Path('orders') / f'{order_id}.json', 'w') as f:
                                    json.dump(order_data, f)
                                state['state'] = None
                                state['data'] = {}
                                return self.stream_openai_response(f"Order placed for {quantity} {product_name}(s). Anything else?", session_id)
                            return self.stream_openai_response(f"Sorry, only {product['quantity']} {product_name}(s) available.", session_id)
                except ValueError:
                    return self.stream_openai_response("Please provide a valid number for the quantity.", session_id)

        # Dynamic Responses
        return self.stream_openai_response(user_input, session_id)

# Instantiate the system
system = CustomerServiceSystem()

# API Endpoints
@app.route('/api/query', methods=['POST'])
def query():
    data = request.get_json()
    if not data or 'input' not in data or 'session_id' not in data:
        return jsonify({"error": "Missing input or session_id"}), 400
    
    user_input = data['input']
    session_id = data['session_id']
    
    def generate_response():
        yield "data: " + json.dumps({"response": ""}) + "\n\n"  # Initial empty chunk
        for chunk in system.handle_query(user_input, session_id):
            yield "data: " + json.dumps({"response": chunk}) + "\n\n"
    
    logger.info("Query processed", extra={"details": user_input, "further": session_id})
    return Response(generate_response(), mimetype='text/event-stream')

@app.route('/api/store_info', methods=['GET'])
def get_store_info():
    return jsonify(system.store_info)

@app.route('/api/products', methods=['GET'])
def get_products():
    return jsonify({"products": system.products})

@app.route('/api/orders', methods=['GET'])
def get_orders():
    orders = []
    for order_file in Path('orders').glob('*.json'):
        with open(order_file, 'r') as f:
            orders.append(json.load(f))
    return jsonify({"orders": orders})

@app.route('/api/meetings', methods=['GET'])
def get_meetings():
    meetings = []
    for meeting_file in Path('meetings').glob('*.json'):
        with open(meeting_file, 'r') as f:
            meetings.append(json.load(f))
    return jsonify({"meetings": meetings})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)