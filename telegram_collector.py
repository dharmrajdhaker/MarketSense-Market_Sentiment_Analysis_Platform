import os
import logging
from datetime import datetime, timezone
import asyncio
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from telethon.tl.types import InputPeerChannel
from utils.config import (
    API_RATE_LIMITS,
    RAW_DATA_DIR,
    TELEGRAM_BOT_TOKEN,
    START_DATE,
    END_DATE
)
import pandas as pd

logger = logging.getLogger(__name__)

class TelegramCollector:
    def __init__(self):
        # Split the bot token to get API ID and hash
        # Format: <api_id>:<api_hash>
        api_id, api_hash = TELEGRAM_BOT_TOKEN.split(':')
        self.client = TelegramClient('anon', api_id, api_hash)
        
    async def _get_channel_messages(self, channel_username, start_date=START_DATE, end_date=END_DATE):
        messages = []
        offset_id = 0
        limit = 100
        reached_end_date = False
        
        try:
            # Convert start_date and end_date to UTC timezone-aware datetime if they aren't already
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=timezone.utc)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=timezone.utc)
                
            logger.info(f"Fetching messages for channel {channel_username} from {start_date} to {end_date}")
            
            # Get channel entity
            try:
                channel = await self.client.get_entity(channel_username)
                logger.info(f"Successfully accessed channel {channel_username}")
            except Exception as e:
                logger.error(f"Error accessing channel {channel_username}: {str(e)}")
                return messages
                
            while not reached_end_date:
                try:
                    # Get messages from channel
                    history = await self.client(GetHistoryRequest(
                        peer=channel,
                        offset_id=offset_id,
                        offset_date=None,
                        add_offset=0,
                        limit=limit,
                        max_id=0,
                        min_id=0,
                        hash=0
                    ))
                    
                    if not history.messages:
                        logger.info(f"No more messages found for channel {channel_username}")
                        break
                        
                    for message in history.messages:
                        # Convert message date to UTC timezone-aware datetime
                        message_date = message.date.replace(tzinfo=timezone.utc)
                        
                        # Stop if we've gone past the end date
                        if message_date > end_date:
                            continue
                            
                        # Stop if we've reached messages before start date
                        if message_date < start_date:
                            logger.info(f"Reached messages before start date for {channel_username}")
                            reached_end_date = True
                            break
                            
                        message_text = message.message or ""
                        if message_text:  # Only add messages with text content
                            messages.append({
                                'date': message_date,
                                'text': message_text,
                                'channel_id': channel.id,
                                'message_id': message.id,
                                'channel': channel_username
                            })
                            logger.info(f"Found message from {channel_username} at {message_date}: {message_text[:50]}...")
                                
                    if history.messages and not reached_end_date:
                        offset_id = history.messages[-1].id
                    await asyncio.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error fetching messages from {channel_username}: {str(e)}")
                    break
                    
        except Exception as e:
            logger.error(f"Unexpected error while fetching messages from {channel_username}: {str(e)}")
            
        return messages
        
    async def collect_data(self, channels, start_date=START_DATE, end_date=END_DATE):
        # Convert start_date and end_date to UTC timezone-aware datetime if they aren't already
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)
            
        all_messages = []
        
        # Start the client
        await self.client.start()
        
        try:
            for channel in channels:
                logger.info(f"Collecting data from channel: {channel}")
                messages = await self._get_channel_messages(channel, start_date, end_date)
                all_messages.extend(messages)
                logger.info(f"Found {len(messages)} messages from {channel}")
                await asyncio.sleep(1)  # Rate limiting between channels
        finally:
            # Always disconnect the client
            await self.client.disconnect()
                
        # Convert to DataFrame and ensure 'text' column exists
        df = pd.DataFrame(all_messages)
        if df.empty:
            logger.warning("No messages were collected from any channels")
            # Create empty DataFrame with required columns
            df = pd.DataFrame(columns=['date', 'text', 'channel_id', 'message_id', 'channel'])
        else:
            # Sort by date
            df = df.sort_values('date')
            
            # Ensure all required columns exist
            required_columns = ['date', 'text', 'channel_id', 'message_id', 'channel']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
            
            # Save raw data
            os.makedirs(RAW_DATA_DIR, exist_ok=True)
            df.to_csv(f"{RAW_DATA_DIR}/telegram_messages_{start_date.date()}_{end_date.date()}.csv", index=False)
            logger.info(f"Saved {len(df)} messages to CSV")
        
        return df
        
    def collect_data_sync(self, channels, start_date=START_DATE, end_date=END_DATE):
        return asyncio.run(self.collect_data(channels, start_date, end_date)) 