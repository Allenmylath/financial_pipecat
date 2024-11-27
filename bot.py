import asyncio
import os
import sys
from loguru import logger
import json
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.clocks.system_clock import SystemClock
from pipecat.frames.frames import StartFrame, EndFrame
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.logger import FrameLogger
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMContext, OpenAILLMService, OpenAILLMContextFrame
from websocket_server import WebsocketServerParams, WebsocketServerTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
from noisereduce_filter import NoisereduceFilter

# Load environment variables
load_dotenv(override=True)

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# Initialize Firebase
firebase_creds = json.loads(os.environ.get('FIREBASE_CREDENTIALS', '{}'))
cred = credentials.Certificate(firebase_creds)
firebase_admin.initialize_app(cred)
db = firestore.client()

class CollectionProcessor:
    def __init__(self, context: OpenAILLMContext, customer_id: str, customer_name: str, overdue_amount: float):
        self.customer_id = customer_id
        self.overdue_amount = overdue_amount
        
        context.add_message(
            {
                "role": "system",
                "content": f"""You are Sarah from ABC Bank's collections department. You're calling regarding an overdue loan payment of ₹{overdue_amount:,.2f}. 
                You're speaking with {customer_name}. Your job is to:

                1. Introduce yourself professionally and verify the customer's identity
                2. Inform them about their overdue payment of ₹{overdue_amount:,.2f}
                3. Emphasize that:
                   - Late payments severely impact their CIBIL score
                   - A low CIBIL score will affect their future loan eligibility
                   - This may impact their ability to get credit cards, loans, or mortgages
                4. Get a firm commitment on a repayment date and amount
                
                Be firm but professional. Use pauses by inserting "-" where needed. Use two question marks for emphasis.
                If the customer tries to avoid committing to a date, remind them about the CIBIL score impact.
                
                Start by introducing yourself and confirming you're speaking with {customer_name}."""
            }
        )
        
        context.set_tools(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "save_repayment_date",
                        "description": "Save the promised repayment date from the customer",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "repayment_date": {
                                    "type": "string",
                                    "description": "The date the customer promises to pay, in YYYY-MM-DD format",
                                },
                                "amount": {
                                    "type": "number",
                                    "description": "The amount the customer promises to pay"
                                }
                            },
                            "required": ["repayment_date", "amount"]
                        },
                    },
                }
            ]
        )

    async def save_repayment_date(self, function_name, tool_call_id, args, llm, context, result_callback):
        logger.info(f"Saving repayment commitment for customer {self.customer_id}: {args}")
        
        # Save to Firebase using the customer_id
        user_ref = db.collection('users').document(self.customer_id)
        user_ref.update({
            "promised_repayment_date": args["repayment_date"],
            "promised_amount": args["amount"],
            "overdue_amount": self.overdue_amount,
            "commitment_made_at": firestore.SERVER_TIMESTAMP,
            "last_contact": firestore.SERVER_TIMESTAMP,
            "contact_type": "phone_call",
            "commitment_status": "pending",
            "payment_history": firestore.ArrayUnion([{
                "date": firestore.SERVER_TIMESTAMP,
                "promised_date": args["repayment_date"],
                "promised_amount": args["amount"],
                "status": "pending"
            }])
        })
        
        context.add_message({
            "role": "system",
            "content": f"""Thank the customer for their commitment to pay ₹{args['amount']:,.2f} by {args['repayment_date']}. Then:
            1. Clearly restate their commitment to pay ₹{args['amount']:,.2f} by {args['repayment_date']}
            2. Remind them that keeping this commitment is crucial for their CIBIL score
            3. Mention that failing to pay by {args['repayment_date']} may result in further action
            4. End the call professionally and courteously
            
            Keep your closing brief but firm."""
        })
        
        context.set_tools([])
        await llm.process_frame(OpenAILLMContextFrame(context), FrameDirection.DOWNSTREAM)

async def main():
    # Initialize WebSocket transport
    transport = WebsocketServerTransport(
        params=WebsocketServerParams(
            host="",
            port=int(os.environ["PORT"]),
            audio_out_enabled=True,
            add_wav_header=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
            audio_in_filter=NoisereduceFilter(),
        )
    )

    # Initialize services
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY")
    )
    
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="829ccd10-f8b3-43cd-b8a0-4aeaa81f3b30",  # Professional female voice
    )

    # Customer details (in practice, these would come from your database)
    customer_id = "chad_bailey"
    customer_name = "Chad Bailey"
    overdue_amount = 25000.00  # Example overdue amount

    # Initialize context and processor
    messages = []
    context = OpenAILLMContext(messages=messages)
    context_aggregator = llm.create_context_aggregator(context)
    collection = CollectionProcessor(context, customer_id, customer_name, overdue_amount)

    # Register function
    llm.register_function("save_repayment_date", collection.save_repayment_date)

    # Initialize frame logger
    fl = FrameLogger("LLM Output")

    # Set up pipeline
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        fl,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    # Create pipeline task
    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=False))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Reset context for new call
        context.messages.clear()
        collection = CollectionProcessor(context, customer_id, customer_name, overdue_amount)
        context_aggregator = llm.create_context_aggregator(context)
        logger.info(f"New client connected. Fresh context created for customer {customer_id}")
        await task.queue_frames([OpenAILLMContextFrame(context)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected for customer {customer_id}")
        await task.queue_frames([EndFrame()])

    # Run the pipeline
    runner = PipelineRunner()
    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())
