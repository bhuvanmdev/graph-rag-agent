#!/usr/bin/env python3
"""
interface.py - Gradio web interface for the RAG application

This module provides the user interface for interacting with the RAG system,
including bulk ticket processing and individual query handling.
"""
import json
import logging
import sqlite3
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

import gradio as gr

from rag import GeminiRAG
from classifier import TicketClassifier
from ticket_manager import SQLiteTicketManager
from models import TicketData

logger = logging.getLogger(__name__)


class GradioInterface:
    """Manages the Gradio web interface"""

    def __init__(self, rag_system: GeminiRAG):
        """Initialize the interface with RAG system"""
        self.rag = rag_system
        self.classifier = TicketClassifier()
        self.ticket_manager = SQLiteTicketManager()
        self.interface = None

        # Load sample tickets
        self.sample_tickets = self._load_sample_tickets()

    def _load_sample_tickets(self) -> List[TicketData]:
        """Load sample tickets from embedded JSON data"""
        try:
            # Sample tickets JSON data (subset from Sample tickets.md)
            sample_tickets_json = """[
                {
                    "id": "TICKET-245",
                    "subject": "Connecting Snowflake to Atlan - required permissions?",
                    "body": "Hi team, we're trying to set up our primary Snowflake production database as a new source in Atlan, but the connection keeps failing. We've tried using our standard service account, but it's not working. Our entire BI team is blocked on this integration for a major upcoming project, so it's quite urgent. Could you please provide a definitive list of the exact permissions and credentials needed on the Snowflake side to get this working? Thanks."
                },
                {
                    "id": "TICKET-246",
                    "subject": "Which connectors automatically capture lineage?",
                    "body": "Hello, I'm new to Atlan and trying to understand the lineage capabilities. The documentation mentions automatic lineage, but it's not clear which of our connectors (we use Fivetran, dbt, and Tableau) support this out-of-the-box. We need to present a clear picture of our data flow to leadership next week. Can you explain how lineage capture differs for these tools?"
                },
                {
                    "id": "TICKET-247",
                    "subject": "Deployment of Atlan agent for private data lake",
                    "body": "Our primary data lake is hosted on-premise within a secure VPC and is not exposed to the internet. We understand we need to use the Atlan agent for this, but the setup instructions are a bit confusing for our security team. This is a critical source for us, and we can't proceed with our rollout until we get this connected. Can you provide a detailed deployment guide or connect us with a technical expert?"
                },
                {
                    "id": "TICKET-248",
                    "subject": "How to surface sample rows and schema changes?",
                    "body": "Hi, we've successfully connected our Redshift cluster, and the assets are showing up. However, my data analysts are asking how they can see sample data or recent schema changes directly within Atlan without having to go back to Redshift. Is this feature available? I feel like I'm missing something obvious."
                },
                {
                    "id": "TICKET-249",
                    "subject": "Exporting lineage view for a specific table",
                    "body": "For our quarterly audit, I need to provide a complete upstream and downstream lineage diagram for our core `fact_orders` table. I can see the lineage perfectly in the UI, but I can't find an option to export this view as an image or PDF. This is a hard requirement from our compliance team and the deadline is approaching fast. Please help!"
                },
                {
                    "id": "TICKET-251",
                    "subject": "Using the Visual Query Builder",
                    "body": "I'm a business analyst and not very comfortable with writing complex SQL. I was excited to see the Visual Query Builder in Atlan, but I'm having trouble figuring out how to join multiple tables and save my query for later use. Is there a tutorial or a quick guide you can point me to?"
                },
                {
                    "id": "TICKET-252",
                    "subject": "Programmatic extraction of lineage",
                    "body": "Our internal data science team wants to build a custom application that analyzes metadata propagation delays. To do this, we need to programmatically extract lineage data from Atlan via an API. Does the API expose lineage information, and if so, could you provide an example of the endpoint and the structure of the response?"
                },
                {
                    "id": "TICKET-254",
                    "subject": "How to create a business glossary and link terms in bulk?",
                    "body": "We are migrating our existing business glossary from a spreadsheet into Atlan. We have over 500 terms. Manually creating each one and linking them to thousands of assets seems impossible. Is there a bulk import feature using CSV or an API to create terms and link them to assets? This is blocking our entire governance initiative."
                },
                {
                    "id": "TICKET-259",
                    "subject": "How does Atlan surface sensitive fields like PII?",
                    "body": "Our security team is evaluating Atlan and their main question is around PII and sensitive data. How does Atlan automatically identify fields containing PII? What are our options to apply tags or masks to these fields once they are identified to prevent unauthorized access?"
                },
                {
                    "id": "TICKET-261",
                    "subject": "Enabling and testing SAML SSO",
                    "body": "We are ready to enable SAML SSO with our Okta instance. However, we are very concerned about disrupting our active users if the configuration is wrong. Is there a way to test the SSO configuration for a specific user or group before we enable it for the entire workspace?"
                }
            ]"""

            tickets_data = json.loads(sample_tickets_json)
            tickets = [TicketData(**ticket) for ticket in tickets_data]
            logger.info(f"Loaded {len(tickets)} sample tickets")
            return tickets

        except Exception as e:
            logger.error(f"Error loading sample tickets: {str(e)}")
            return []

    def process_bulk_tickets(self) -> Tuple[str, str]:
        """
        Process all sample tickets in bulk

        Returns:
            Tuple of (results_html, summary_text)
        """
        if not self.sample_tickets:
            return "No sample tickets to process.", "No tickets processed."

        logger.info(f"Processing {len(self.sample_tickets)} tickets in bulk...")

        results = []
        rag_count = 0
        human_count = 0

        for ticket in self.sample_tickets:
            try:
                # Classify the ticket
                classification = self.classifier.classify_ticket(ticket)

                # Determine if RAG should be used
                use_rag = self.classifier.should_use_rag(classification.topic)

                rag_answer = None
                if use_rag:
                    # Generate RAG answer
                    query = f"{ticket.subject} {ticket.body}"
                    rag_answer = self.rag.answer_query(query, top_k=10)
                    rag_count += 1
                else:
                    human_count += 1

                # Store in database
                self.ticket_manager.store_ticket(ticket, classification, rag_answer)

                # Format result for display
                result = {
                    'ticket_id': ticket.id,
                    'subject': ticket.subject,
                    'topic': classification.topic.value,
                    'sentiment': classification.sentiment.value,
                    'priority': classification.priority.value,
                    'reasoning': classification.reasoning,
                    'use_rag': use_rag,
                    'rag_answer': rag_answer.answer if rag_answer else "Routed to human agent",
                    'sources': rag_answer.sources if rag_answer else []
                }
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing ticket {ticket.id}: {str(e)}")
                continue

        # Generate HTML results table
        results_html = self._format_results_html(results)

        # Generate summary
        summary = f"""
        **Bulk Processing Complete!**

        📊 **Summary:**
        - Total tickets processed: {len(results)}
        - RAG-answered tickets: {rag_count}
        - Human-routed tickets: {human_count}

        🤖 **RAG Topics:** How-to, Product, Best practices, API/SDK, SSO
        👥 **Human Topics:** Connector, Lineage, Glossary, Sensitive data
        """

        return results_html, summary

    def _format_results_html(self, results: List[Dict]) -> str:
        """Format classification results as HTML table"""
        if not results:
            return "<p>No results to display.</p>"

        html = """
        <div class="results-table">
        <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
        <thead class="table-header">
            <tr>
                <th style="color: black;">Ticket ID</th>
                <th style="color: black;">Subject</th>
                <th style="color: black;">Topic</th>
                <th style="color: black;">Sentiment</th>
                <th style="color: black;">Priority</th>
                <th style="color: black;">Handling</th>
                <th style="color: black;">Response/Action</th>
            </tr>
        </thead>
        <tbody class="table-body">
        """

        for result in results:
            # Determine row class based on handling type
            row_class = "rag-row" if result['use_rag'] else "human-row"
            handling = "🤖 RAG Answer" if result['use_rag'] else "👥 Human Review"
            handling_class = "handling-rag" if result['use_rag'] else "handling-human"

            # Determine priority class
            priority = result['priority']
            if "P0" in priority or "High" in priority:
                priority_class = "priority-high"
            elif "P1" in priority or "Medium" in priority:
                priority_class = "priority-medium"
            else:
                priority_class = "priority-low"

            # Truncate response for display
            response = result['rag_answer']
            if len(response) > 200:
                truncated_response = response[:200] + "..."
                full_response = response.replace('\n', '<br>').replace('"', '&quot;').replace("'", "&#39;")
                response_html = f'''
                <div class="expandable-text" onclick="
                    var truncated = this.querySelector('.truncated-text');
                    var full = this.querySelector('.full-text');
                    var indicator = this.querySelector('.expand-indicator');
                    if (this.classList.contains('expanded')) {{
                        this.classList.remove('expanded');
                        indicator.innerHTML = ' [Click to expand]';
                    }} else {{
                        this.classList.add('expanded');
                        indicator.innerHTML = ' [Click to collapse]';
                    }}
                ">
                    <span class="truncated-text" style="color: #333333;">{truncated_response}</span>
                    <span class="full-text" style="color: #333333;">{full_response}</span>
                    <span class="expand-indicator"> [Click to expand]</span>
                </div>
                '''
            else:
                response_html = f'<span style="color: #333333;">{response}</span>'

            html += f"""
            <tr class="{row_class}">
                <td style="color: black;">{result['ticket_id']}</td>
                <td style="max-width: 200px; word-wrap: break-word; color: black;">{result['subject']}</td>
                <td style="color: black;">{result['topic']}</td>
                <td style="color: black;">{result['sentiment']}</td>
                <td class="{priority_class}" style="color: black;">{result['priority']}</td>
                <td class="{handling_class}" style="color: black;">{handling}</td>
                <td style="max-width: 300px; word-wrap: break-word; color: black;">{response_html}</td>
            </tr>
            """

        html += """
        </tbody>
        </table>
        </div>
        """

        return html

    def process_single_ticket(self, subject: str, body: str) -> Tuple[str, str, str, str, str, bool, Optional[Any]]:
        """
        Process a single new ticket

        Returns:
            Tuple of (backend_view, frontend_view, sources_info, routing_info, ticket_id, use_rag, rag_answer)
        """
        if not subject or not body:
            return "Please provide both subject and body.", "", "", "", "", False, None

        # Create ticket dictionary
        ticket = TicketData(
            id=f"NEW-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            subject=subject,
            body=body
        )

        try:
            # Classify the ticket
            classification = self.classifier.classify_ticket(ticket)

            # Determine if RAG should be used
            use_rag = self.classifier.should_use_rag(classification.topic)

            # Format backend view (internal analysis)
            backend_view = f"""
### 🔍 **BACKEND VIEW** (Internal Analysis)

**📊 Classification Results:**
- **Topic:** `{classification.topic.value}`
- **Sentiment:** `{classification.sentiment.value}`
- **Priority:** `{classification.priority.value}`

**🧠 AI Reasoning:**
> {classification.reasoning}

**🎯 Routing Decision:** {'🤖 RAG Pipeline' if use_rag else '👥 Human Agent'}
            """

            if use_rag:
                # Generate RAG answer
                query = f"Subject: {subject}\nBody: {body}"
                rag_answer = self.rag.answer_query(query, top_k=5)
                # Store with RAG answer
                self.ticket_manager.store_ticket(ticket, classification, rag_answer)

                # Format frontend view (customer response)
                frontend_view = f"""
### 💬 **FRONTEND VIEW** (Customer Response)

{rag_answer.answer}
"""         
            
                # Format sources information
                sources_info = ""
                if rag_answer.sources:
                    sources_info = f"""
### 📚 **SOURCES USED**

{"<br>".join([f"{num+1}. 📄 {source}" for num,source in enumerate(rag_answer.sources)])}
                """

                # Format routing information
                routing_info = f"""
### ✅ **RAG SYSTEM RESPONSE**

**Status:** Automatically answered by AI<br>
**Processing Time:** Real-time<br>
**Confidence:** High (based on documentation sources)<br>
**Next Steps:** Response sent to customer
                """

                if use_rag:
                    frontend_view += f"""            
This issue was classified as a **{classification.topic.value}** topic and has been automatically addressed by our AI system. If you are not satisfied with this response, please reply click on the ask human button below.
"""

            else:
                # For human routing
                rag_answer = None
                self.ticket_manager.store_ticket(ticket, classification, None)

                frontend_view = f"""
### 💬 **FRONTEND VIEW** (Customer Response)

Thank you for your inquiry. This issue has been classified as a **{classification.topic.value}** topic and requires human assistance. Our support team will review your request and get back to you within our standard SLA timeframe.

**Topic:** {classification.topic.value}

**Priority:** {classification.priority.value}

**Next Steps:** A human agent will be assigned to your case shortly.

"""

                sources_info = ""

                routing_info = f"""
### 👥 **HUMAN AGENT ROUTING**

**Status:** Ticket routed to human agent

**Reason:** Topic requires specialized human expertise

**Queue:** Added to human review queue

**Expected Response:** Within SLA timeframe

"""

            return backend_view, frontend_view, sources_info, routing_info, ticket.id, use_rag, rag_answer

        except Exception as e:
            logger.error(f"Error processing single ticket: {str(e)}")
            error_msg = f"**Error processing ticket:** {str(e)}"
            return error_msg, error_msg, "", "", "", False, None

    def request_human_review(self, ticket_id: str, original_question: str, rag_answer: str) -> Tuple[str, bool]:
        """
        Handle request for human review
        
        Returns:
            Tuple of (status_message, button_interactive)
        """
        if not ticket_id:
            return "No ticket ID available.", True
            
        # Check if human review already requested
        existing_status = self.ticket_manager.check_human_review_status(ticket_id)
        if existing_status:
            return f"""
### ✅ **HUMAN REVIEW STATUS**

**Status:** {existing_status.title()}
**Message:** {"A human agent is already reviewing your request." if existing_status == 'pending' else "Your request is being processed by our team."}
**Next Steps:** You will receive a response from our human agent soon.
            """, False
        
        # Store the human review request
        success = self.ticket_manager.request_human_review(ticket_id, original_question, rag_answer)
        
        if success:
            return """
### ✅ **HUMAN REVIEW REQUESTED**

**Status:** Successfully submitted for human review
**Queue:** Your request has been added to our human agent queue
**Expected Response:** A human agent will review your case and respond within our standard SLA timeframe
**Next Steps:** You will receive a personalized response addressing your specific concerns

Thank you for your patience while our expert team prepares a comprehensive response.
            """, False
        else:
            return "Failed to submit human review request. Please try again.", True

    def ask_ai(self, ticket_id: str, question: str) -> Tuple[str, bool, str, str, str]:
        """
        Generate AI response for a ticket that was initially routed to human
        
        Returns:
            Tuple of (status_message, button_interactive, new_frontend, new_sources, new_routing)
        """
        if not ticket_id or not question:
            return "No ticket information available.", True, "", "", ""

        # Generate RAG answer
        rag_answer = self.rag.answer_query(question, top_k=5)

        # Update the ticket with RAG answer
        self.ticket_manager.update_ticket_answer(ticket_id, rag_answer)

        # Format new frontend
        new_frontend = f"""
### 💬 **FRONTEND VIEW** (Customer Response)

{rag_answer.answer}

This issue was addressed by our AI system. If you are not satisfied with this response, please reply to request human assistance.
"""

        new_sources = ""
        if rag_answer.sources:
            new_sources = f"""
### 📚 **SOURCES USED**

{"<br>".join([f"{num}. 📄 {source}" for num,source in enumerate(rag_answer.sources)])}
"""

        new_routing = f"""
### ✅ **AI RESPONSE GENERATED**

**Status:** Manually invoked AI response
**Processing Time:** Real-time
**Next Steps:** Response sent to customer
"""

        return f"""
### ✅ **AI Response Generated**

**Status:** RAG answer has been generated for this ticket
**Response:** {rag_answer.answer[:100]}...
""", False, new_frontend, new_sources, new_routing

    def handle_button_click(self, use_rag: bool, ticket_id: str, question: str, rag_answer: str) -> Tuple[str, Any, str, str, str]:
        """
        Handle button click based on use_rag flag
        
        Returns:
            Tuple of (status_message, button_update, frontend_update, sources_update, routing_update)
        """
        if use_rag:
            # Request human review
            status, interactive = self.request_human_review(ticket_id, question, rag_answer)
            return status, gr.update(interactive=interactive), "", "", ""
        else:
            # Generate AI response
            status, interactive, new_frontend, new_sources, new_routing = self.ask_ai(ticket_id, question)
            return status, gr.update(interactive=interactive), new_frontend, new_sources, new_routing

    def _get_rag_answer_for_ticket(self, ticket_id: str) -> str:
        """Get the RAG answer for a specific ticket"""
        if not ticket_id:
            return ""
        
        ticket_data = self.ticket_manager.get_ticket(ticket_id)
        if ticket_data and ticket_data.get('answer'):
            return ticket_data['answer']['answer']
        return ""

    def load_database_contents(self) -> Tuple[str, str, str, str, str]:
        """
        Load and format all database contents for display
        
        Returns:
            Tuple of (stats_html, tickets_html, classifications_html, answers_html, reviews_html)
        """
        try:
            # Get database statistics
            stats = self.ticket_manager.get_ticket_stats()
            db_info = self.ticket_manager.get_database_info()
            
            # Format statistics
            stats_html = f"""
            <div class="db-stats-container">
                <h3>📊 Database Statistics</h3>
                <div class="stats-grid">
                    <div class="stat-card">
                        <h4>📄 Total Tickets</h4>
                        <span class="stat-number">{stats['total_tickets']}</span>
                    </div>
                    <div class="stat-card">
                        <h4>🤖 RAG Answered</h4>
                        <span class="stat-number">{stats['rag_tickets']}</span>
                    </div>
                    <div class="stat-card">
                        <h4>👥 Human Routed</h4>
                        <span class="stat-number">{stats['human_tickets']}</span>
                    </div>
                    <div class="stat-card">
                        <h4>💾 DB Size</h4>
                        <span class="stat-number">{db_info['db_size_mb']} MB</span>
                    </div>
                </div>
                <div class="topics-distribution">
                    <h4>📈 Topic Distribution</h4>
                    <ul>
                        {"".join([f"<li><strong>{topic}:</strong> {count}</li>" for topic, count in stats['topics'].items()])}
                    </ul>
                </div>
            </div>
            """
            
            # Load all tables data
            tickets_html = self._load_tickets_table()
            classifications_html = self._load_classifications_table()
            answers_html = self._load_answers_table()
            reviews_html = self._load_human_reviews_table()
            
            return stats_html, tickets_html, classifications_html, answers_html, reviews_html
            
        except Exception as e:
            error_msg = f"<p style='color: red;'>Error loading database: {str(e)}</p>"
            return error_msg, error_msg, error_msg, error_msg, error_msg

    def _load_tickets_table(self) -> str:
        """Load and format tickets table"""
        with sqlite3.connect(self.ticket_manager.db_path) as conn:
            cursor = conn.execute('''
                SELECT id, subject, body, created_at, updated_at
                FROM tickets
                ORDER BY created_at DESC
            ''')
            
            rows = cursor.fetchall()
            
            if not rows:
                return "<p>No tickets found in database.</p>"
            
            html = """
            <div class="db-table-container">
                <table class="db-table">
                    <thead>
                        <tr>
                            <th>Ticket ID</th>
                            <th>Subject</th>
                            <th>Body Preview</th>
                            <th>Created</th>
                            <th>Updated</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for row in rows:
                ticket_id, subject, body, created_at, updated_at = row
                body_preview = body[:100] + "..." if len(body) > 100 else body
                
                html += f"""
                <tr>
                    <td><code>{ticket_id}</code></td>
                    <td>{subject}</td>
                    <td>{body_preview}</td>
                    <td>{created_at}</td>
                    <td>{updated_at}</td>
                </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            """
            
            return html

    def _load_classifications_table(self) -> str:
        """Load and format classifications table"""
        with sqlite3.connect(self.ticket_manager.db_path) as conn:
            cursor = conn.execute('''
                SELECT c.ticket_id, c.topic, c.sentiment, c.priority, c.reasoning, c.use_rag, c.created_at,
                       t.subject
                FROM classifications c
                JOIN tickets t ON c.ticket_id = t.id
                ORDER BY c.created_at DESC
            ''')
            
            rows = cursor.fetchall()
            
            if not rows:
                return "<p>No classifications found in database.</p>"
            
            html = """
            <div class="db-table-container">
                <table class="db-table">
                    <thead>
                        <tr>
                            <th>Ticket ID</th>
                            <th>Subject</th>
                            <th>Topic</th>
                            <th>Sentiment</th>
                            <th>Priority</th>
                            <th>Use RAG</th>
                            <th>Reasoning</th>
                            <th>Classified</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for row in rows:
                ticket_id, topic, sentiment, priority, reasoning, use_rag, created_at, subject = row
                reasoning_preview = reasoning[:50] + "..." if len(reasoning) > 50 else reasoning
                use_rag_display = "🤖 Yes" if use_rag else "👥 No"
                
                # Add priority styling
                priority_class = ""
                if "P0" in priority or "High" in priority:
                    priority_class = "priority-high"
                elif "P1" in priority or "Medium" in priority:
                    priority_class = "priority-medium"
                else:
                    priority_class = "priority-low"
                
                html += f"""
                <tr>
                    <td><code>{ticket_id}</code></td>
                    <td>{subject[:30]}...</td>
                    <td><span class="topic-badge">{topic}</span></td>
                    <td><span class="sentiment-badge">{sentiment}</span></td>
                    <td><span class="{priority_class}">{priority}</span></td>
                    <td>{use_rag_display}</td>
                    <td>{reasoning_preview}</td>
                    <td>{created_at}</td>
                </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            """
            
            return html

    def _load_answers_table(self) -> str:
        """Load and format answers table"""
        with sqlite3.connect(self.ticket_manager.db_path) as conn:
            cursor = conn.execute('''
                SELECT a.ticket_id, a.answer, a.sources, a.chunks_used, a.created_at,
                       t.subject
                FROM answers a
                JOIN tickets t ON a.ticket_id = t.id
                ORDER BY a.created_at DESC
            ''')
            
            rows = cursor.fetchall()
            
            if not rows:
                return "<p>No RAG answers found in database.</p>"
            
            html = """
            <div class="db-table-container">
                <table class="db-table">
                    <thead>
                        <tr>
                            <th>Ticket ID</th>
                            <th>Subject</th>
                            <th>Answer Preview</th>
                            <th>Sources Count</th>
                            <th>Chunks Used</th>
                            <th>Generated</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for row in rows:
                ticket_id, answer, sources_json, chunks_used, created_at, subject = row
                answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
                
                try:
                    sources = json.loads(sources_json) if sources_json else []
                    sources_count = len(sources)
                except:
                    sources_count = 0
                
                html += f"""
                <tr>
                    <td><code>{ticket_id}</code></td>
                    <td>{subject[:30]}...</td>
                    <td>{answer_preview}</td>
                    <td>{sources_count}</td>
                    <td>{chunks_used}</td>
                    <td>{created_at}</td>
                </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            """
            
            return html

    def _load_human_reviews_table(self) -> str:
        """Load and format human reviews table"""
        with sqlite3.connect(self.ticket_manager.db_path) as conn:
            cursor = conn.execute('''
                SELECT hr.id, hr.ticket_id, hr.original_question, hr.rag_answer, 
                       hr.requested_at, hr.status, hr.human_response, hr.responded_at,
                       t.subject
                FROM human_reviews hr
                JOIN tickets t ON hr.ticket_id = t.id
                ORDER BY hr.requested_at DESC
            ''')
            
            rows = cursor.fetchall()
            
            if not rows:
                return "<p>No human review requests found in database.</p>"
            
            html = """
            <div class="db-table-container">
                <table class="db-table">
                    <thead>
                        <tr>
                            <th>Review ID</th>
                            <th>Ticket ID</th>
                            <th>Subject</th>
                            <th>Question Preview</th>
                            <th>Status</th>
                            <th>Requested</th>
                            <th>Responded</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for row in rows:
                review_id, ticket_id, original_question, rag_answer, requested_at, status, human_response, responded_at, subject = row
                question_preview = original_question[:80] + "..." if len(original_question) > 80 else original_question
                
                # Add status styling
                status_class = ""
                if status == "pending":
                    status_class = "status-pending"
                elif status == "in_progress":
                    status_class = "status-progress"
                elif status == "completed":
                    status_class = "status-completed"
                
                html += f"""
                <tr>
                    <td>{review_id}</td>
                    <td><code>{ticket_id}</code></td>
                    <td>{subject[:30]}...</td>
                    <td>{question_preview}</td>
                    <td><span class="{status_class}">{status.title()}</span></td>
                    <td>{requested_at}</td>
                    <td>{responded_at or 'N/A'}</td>
                </tr>
                """
            
            html += """
                    </tbody>
                </table>
            </div>
            """
            
            return html

    def clear_database(self) -> Tuple[str, str, str, str, str]:
        """
        Clear all database contents
        
        Returns:
            Tuple of (stats_html, tickets_html, classifications_html, answers_html, reviews_html)
        """
        try:
            self.ticket_manager.clear_all_tickets()
            
            # Clear human reviews table as well
            with sqlite3.connect(self.ticket_manager.db_path) as conn:
                conn.execute('DELETE FROM human_reviews')
                conn.commit()
            
            empty_msg = "<p>Database cleared. All data has been removed.</p>"
            stats_html = """
            <div class="db-stats-container">
                <h3>📊 Database Statistics</h3>
                <p style="color: #666;">Database has been cleared. All data removed.</p>
            </div>
            """
            
            return stats_html, empty_msg, empty_msg, empty_msg, empty_msg
            
        except Exception as e:
            error_msg = f"<p style='color: red;'>Error clearing database: {str(e)}</p>"
            return error_msg, error_msg, error_msg, error_msg, error_msg

    def process_query(self, query: str, top_k: int) -> Tuple[str, str, str]:
        """
        Process user query and return formatted results

        Returns:
            Tuple of (answer, sources, debug_info)
        """
        if not query:
            return "Please enter a question.", "", ""

        logger.info(f"Processing query: {query}")

        # Get answer from RAG system
        result = self.rag.answer_query(query, top_k)

        # Format answer
        answer = result.answer

        # Format sources
        if result.sources:
            sources = "📚 **Sources Used:**\n"
            for i, source in enumerate(result.sources, 1):
                sources += f"{i}. {source}\n"
        else:
            sources = "No sources available."

        # Format debug information
        debug_info = f"**Retrieval Statistics:**\n"
        debug_info += f"- Chunks retrieved: {result.chunks_used}\n"
        if result.relevance_scores:
            avg_score = sum(result.relevance_scores) / len(result.relevance_scores)
            debug_info += f"- Average relevance score: {avg_score:.3f}\n"
            debug_info += f"- Top relevance score: {max(result.relevance_scores):.3f}\n"

        return answer, sources, debug_info

    def create_interface(self) -> gr.Blocks:
        """Create and configure the Gradio interface with tabs"""

        # Get database statistics
        try:
            stats = self.rag.retriever.get_database_stats()
            db_info = f"📊 **Knowledge Base Statistics:**\n"
            db_info += f"- Total pages: {stats['pages']}\n"
            db_info += f"- Total chunks: {stats['chunks']}\n"
            db_info += f"- Page links: {stats['links']}"
        except Exception as e:
            db_info = f"📊 **Knowledge Base Statistics:**\nError loading stats: {str(e)}"

        with gr.Blocks(
            title="Customer Support Copilot - RAG System",
            css="""
            .container {
                max-width: 1400px;
                margin: auto;
            }
            .info-box {
                padding: 25px;
                margin: 20px 0;
                border-radius: 15px;
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 50%, #43e97b 100%);
                color: #ffffff;
                border-left: 6px solid #ffffff;
                box-shadow: 0 8px 25px rgba(79, 172, 254, 0.4);
                font-weight: 600;
                font-size: 14px;
                line-height: 1.6;
                text-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
                position: relative;
                overflow: hidden;
            }
            .info-box::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
                pointer-events: none;
            }
            .info-box strong {
                color: #ffffff;
                font-size: 1.2em;
                font-weight: 700;
                text-shadow: 0 2px 4px rgba(0, 0, 0, 0.4);
                display: block;
                margin-bottom: 10px;
            }
            .info-box * {
                position: relative;
                z-index: 1;
            }
            .results-table {
                max-height: 600px;
                overflow-y: auto;
                border: 2px solid #e0e0e0;
                border-radius: 12px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            }
            .rag-row {
                background: linear-gradient(135deg, #e8f5e8 0%, #f3e5f5 100%);
                transition: all 0.3s ease;
                border-left: 4px solid #4caf50;
            }
            .rag-row:hover {
                background: linear-gradient(135deg, #c8e6c9 0%, #e1bee7 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
            }
            .human-row {
                background: linear-gradient(135deg, #fff3e0 0%, #fce4ec 100%);
                transition: all 0.3s ease;
                border-left: 4px solid #ff9800;
            }
            .human-row:hover {
                background: linear-gradient(135deg, #ffe0b2 0%, #f8bbd9 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(255, 152, 0, 0.3);
            }
            .table-header {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 50%, #43e97b 100%);
                color: white;
                font-weight: 700;
                position: sticky;
                top: 0;
                z-index: 10;
                text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
            }
            .table-header th {
                border: none;
                padding: 12px 8px;
                text-align: left;
                font-size: 13px;
            }
            .table-body td {
                border: 1px solid #e0e0e0;
                padding: 12px 10px;
                font-size: 13px;
                vertical-align: top;
                color: #333333;
                font-weight: 500;
            }
            .priority-high {
                color: #d32f2f;
                font-weight: bold;
                background-color: rgba(211, 47, 47, 0.1);
                padding: 2px 6px;
                border-radius: 4px;
            }
            .priority-medium {
                color: #f57c00;
                font-weight: bold;
                background-color: rgba(245, 124, 0, 0.1);
                padding: 2px 6px;
                border-radius: 4px;
            }
            .priority-low {
                color: #388e3c;
                font-weight: bold;
                background-color: rgba(56, 142, 60, 0.1);
                padding: 2px 6px;
                border-radius: 4px;
            }
            .handling-rag {
                color: #1976d2;
                font-weight: bold;
                background-color: rgba(25, 118, 210, 0.1);
                padding: 2px 6px;
                border-radius: 4px;
            }
            .handling-human {
                color: #616161;
                font-weight: bold;
                background-color: rgba(97, 97, 97, 0.1);
                padding: 2px 6px;
                border-radius: 4px;
            }
            .ask-human-btn {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
                border: none;
                color: white;
                font-weight: 600;
                border-radius: 8px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
            }
            .ask-human-btn:hover {
                background: linear-gradient(135deg, #ff5252 0%, #d32f2f 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
            }
            .ask-human-btn:disabled {
                background: #cccccc;
                color: #666666;
                cursor: not-allowed;
                transform: none;
                box-shadow: none;
            }
            .db-stats-container {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 12px;
                margin: 20px 0;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 15px 0;
            }
            .stat-card {
                background: rgba(255, 255, 255, 0.2);
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                backdrop-filter: blur(10px);
            }
            .stat-card h4 {
                margin: 0 0 10px 0;
                font-size: 14px;
                opacity: 0.9;
            }
            .stat-number {
                font-size: 24px;
                font-weight: bold;
                display: block;
            }
            .topics-distribution {
                margin-top: 20px;
                background: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 8px;
            }
            .topics-distribution ul {
                margin: 10px 0;
                padding-left: 20px;
            }
            .topics-distribution li {
                margin: 5px 0;
            }
            .db-table-container {
                max-height: 500px;
                overflow-y: auto;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin: 10px 0;
            }
            .db-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
            }
            .db-table thead {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                position: sticky;
                top: 0;
                z-index: 10;
            }
            .db-table th, .db-table td {
                border: 1px solid #e0e0e0;
                padding: 8px;
                text-align: left;
                word-wrap: break-word;
                color: #333333;
            }
            .db-table code {
                background-color: transparent !important;
                color: #333333 !important;
                padding: 2px 4px;
                border-radius: 3px;
                font-family: monospace;
                border: 1px solid #ddd;
            }
            .db-table th {
                font-weight: bold;
                font-size: 11px;
            }
            .db-table tbody tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .db-table tbody tr:nth-child(odd) {
                background-color: #ffffff;
            }
            .db-table tbody tr:hover {
                background-color: #e3f2fd !important;
            }
            .topic-badge {
                background-color: #e3f2fd;
                color: #1976d2;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
            }
            .sentiment-badge {
                background-color: #e8f5e8;
                color: #2e7d32;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
            }
            .status-pending {
                background-color: #fff3e0;
                color: #f57c00;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
            }
            .status-progress {
                background-color: #e3f2fd;
                color: #1976d2;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
            }
            .status-completed {
                background-color: #e8f5e8;
                color: #2e7d32;
                padding: 2px 6px;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
            }
            .expandable-text {
                cursor: pointer;
                position: relative;
                color: #333333;
            }
            .expandable-text:hover {
                background-color: #f0f0f0;
                border-radius: 4px;
            }
            .expandable-text.expanded {
                background-color: #e8f5e8;
                border-radius: 4px;
                padding: 5px;
            }
            .expand-indicator {
                color: #1976d2;
                font-weight: bold;
                font-size: 12px;
            }
            .full-text {
                display: none;
            }
            .expandable-text.expanded .truncated-text {
                display: none;
            }
            .expandable-text.expanded .full-text {
                display: inline;
            }
            """
        ) as interface:

            gr.Markdown(
                """
                # 🎫 Customer Support Copilot - RAG System

                **Atlan Customer Support AI Pipeline Demo**

                This system demonstrates automated ticket classification and intelligent response generation.
                Choose a tab below to explore different functionalities.
                """,
                elem_classes=["container"]
            )

            # Database statistics display
            gr.Markdown(db_info, elem_classes=["info-box"])

            with gr.Tabs():
                # Tab 1: Bulk Ticket Classification Dashboard
                with gr.TabItem("📊 Bulk Ticket Classification Dashboard"):
                    gr.Markdown("""
                    ### 🔄 Bulk Processing of Sample Tickets

                    This dashboard processes all sample tickets in bulk, demonstrating the complete AI pipeline:
                    - **Classification**: Topic, Sentiment, Priority analysis
                    - **Routing Decision**: RAG vs Human assignment
                    - **Response Generation**: Automated answers for suitable topics

                    **RAG Topics** (Auto-answered): How-to, Product, Best practices, API/SDK, SSO
                    **Human Topics** (Routed): Connector, Lineage, Glossary, Sensitive data
                    """)

                    with gr.Row():
                        with gr.Column(scale=1):
                            process_bulk_btn = gr.Button("🚀 Process All Tickets", variant="primary", size="lg")

                        with gr.Column(scale=3):
                            bulk_results = gr.HTML(value="<p>Click 'Process All Tickets' to start bulk processing.</p>")
                            bulk_summary = gr.Markdown(value="")

                    # Bulk processing event
                    process_bulk_btn.click(
                        fn=self.process_bulk_tickets,
                        outputs=[bulk_results, bulk_summary]
                    )

                # Tab 2: Interactive AI Agent
                with gr.TabItem("🤖 Interactive AI Agent"):
                    gr.Markdown("""
                    ### 💬 Submit New Ticket for AI Processing

                    Submit a new customer support ticket to see the complete AI pipeline in action:
                    1. **Backend Analysis**: Classification and routing decision (internal view)
                    2. **Frontend Response**: What the customer sees
                    3. **Sources & Routing**: Documentation sources used and processing method
                    """)

                    with gr.Row():
                        # Input section
                        with gr.Column(scale=1):
                            new_subject = gr.Textbox(
                                label="📋 Ticket Subject",
                                placeholder="Enter ticket subject...",
                                lines=1
                            )
                            new_body = gr.Textbox(
                                label="📝 Ticket Body", 
                                placeholder="Enter ticket description...",
                                lines=5
                            )
                            submit_ticket_btn = gr.Button("🎫 Process Ticket", variant="primary", size="lg")

                        # Output section - organized as horizontal tabs
                        with gr.Column(scale=2):
                            with gr.Tabs():
                                with gr.TabItem("💬 Frontend View"):
                                    frontend_output = gr.Markdown(
                                        value="Submit a ticket to see the customer response...",
                                        elem_classes=["frontend-view"]
                                    )
                                
                                with gr.TabItem("🔍 Backend Analysis"):
                                    backend_output = gr.Markdown(
                                        value="Submit a ticket to see the internal analysis...",
                                        elem_classes=["backend-view"]
                                    )
                                
                                with gr.TabItem("🤖 RAG Response"):
                                    routing_output = gr.Markdown(
                                        value="Submit a ticket to see the processing method...",
                                        elem_classes=["routing-view"]
                                    )
                                
                                with gr.TabItem("📚 Sources"):
                                    sources_output = gr.Markdown(
                                        value="Submit a ticket to see the sources used...",
                                        elem_classes=["sources-view"]
                                    )

                    # Human Review Section
                    with gr.Row():
                        with gr.Column(scale=1):
                            ask_human_btn = gr.Button(
                                "🤖 Ask AI / 👥 Ask Human", 
                                variant="secondary", 
                                size="lg",
                                interactive=False,
                                visible=False,
                                elem_classes=["ask-human-btn"]
                            )
                            
                        with gr.Column(scale=2):
                            human_review_status = gr.Markdown(
                                value="",
                                visible=False
                            )

                    # Hidden state to store current ticket info
                    current_ticket_id = gr.State("")
                    current_question = gr.State("")
                    current_rag_answer = gr.State("")
                    current_use_rag = gr.State(False)

                    # New ticket processing event
                    submit_ticket_btn.click(
                        fn=self.process_single_ticket,
                        inputs=[new_subject, new_body],
                        outputs=[backend_output, frontend_output, sources_output, routing_output, current_ticket_id, current_use_rag, current_rag_answer]
                    ).then(
                        fn=lambda ticket_id, subject, body: (
                            f"{subject}\n\n{body}",  # current_question
                            "",  # current_rag_answer (will be set below)
                        ),
                        inputs=[current_ticket_id, new_subject, new_body],
                        outputs=[current_question, current_rag_answer]
                    ).then(
                        fn=lambda ticket_id: self._get_rag_answer_for_ticket(ticket_id) if ticket_id else "",
                        inputs=[current_ticket_id],
                        outputs=[current_rag_answer]
                    ).then(
                        fn=lambda use_rag, ticket_id: gr.update(
                            value="👥 Ask Human" if use_rag else "🤖 Ask AI",
                            interactive=bool(ticket_id),
                            visible=bool(ticket_id)
                        ),
                        inputs=[current_use_rag, current_ticket_id],
                        outputs=[ask_human_btn]
                    ).then(
                        fn=lambda: gr.update(visible=False),
                        outputs=[human_review_status]
                    )

                    # Dynamic button click event (handles both Ask Human and Ask AI)
                    ask_human_btn.click(
                        fn=self.handle_button_click,
                        inputs=[current_use_rag, current_ticket_id, current_question, current_rag_answer],
                        outputs=[human_review_status, ask_human_btn, frontend_output, sources_output, routing_output]
                    ).then(
                        fn=lambda: gr.update(visible=True),
                        outputs=[human_review_status]
                    )

                # Tab 3: Ticket SQL Database
                with gr.TabItem("🗄️ Ticket SQL DB"):
                    gr.Markdown("""
                    ### 🗄️ Database Contents Viewer

                    View all stored tickets, classifications, answers, and human review requests in the SQLite database.
                    """)

                    with gr.Row():
                        refresh_db_btn = gr.Button("🔄 Refresh Database", variant="primary", size="lg")
                        clear_db_btn = gr.Button("🗑️ Clear All Data", variant="stop", size="lg")

                    # Database statistics
                    db_stats = gr.HTML(value="<p>Click 'Refresh Database' to load data.</p>")

                    with gr.Tabs():
                        with gr.TabItem("🎫 Tickets"):
                            tickets_table = gr.HTML(value="<p>No tickets data loaded.</p>")
                        
                        with gr.TabItem("📊 Classifications"):
                            classifications_table = gr.HTML(value="<p>No classifications data loaded.</p>")
                        
                        with gr.TabItem("🤖 RAG Answers"):
                            answers_table = gr.HTML(value="<p>No answers data loaded.</p>")
                        
                        with gr.TabItem("👥 Human Reviews"):
                            human_reviews_table = gr.HTML(value="<p>No human reviews data loaded.</p>")

                    # Database events
                    refresh_db_btn.click(
                        fn=self.load_database_contents,
                        outputs=[db_stats, tickets_table, classifications_table, answers_table, human_reviews_table]
                    )

                    clear_db_btn.click(
                        fn=self.clear_database,
                        outputs=[db_stats, tickets_table, classifications_table, answers_table, human_reviews_table]
                    )

                # Tab 4: Knowledge Base Query (Original functionality)
                with gr.TabItem("📚 Knowledge Base Query"):
                    gr.Markdown("""
                    ### 🔍 Direct Knowledge Base Search

                    Query the RAG knowledge base directly for questions about Atlan's documentation.
                    """)

                    with gr.Row():
                        query_input = gr.Textbox(
                            label="Question",
                            placeholder="Ask a question about Atlan...",
                            lines=2
                        )
                        search_btn = gr.Button("🔍 Search", variant="primary")

                    # Output areas for knowledge base query
                    with gr.Row():
                        kb_answer_output = gr.Markdown(value="")
                        kb_sources_output = gr.Markdown(value="")

                    # Advanced options (collapsible)
                    with gr.Accordion("🔧 Advanced Options", open=False):
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of chunks to retrieve"
                        )
                        show_debug = gr.Checkbox(label="Show debug information", value=False)
                        kb_debug_output = gr.Markdown(value="", visible=False)

                    # Knowledge base query events
                    search_btn.click(
                        fn=self.process_query,
                        inputs=[query_input, top_k_slider],
                        outputs=[kb_answer_output, kb_sources_output, kb_debug_output]
                    )

                    # Allow Enter key to submit
                    query_input.submit(
                        fn=self.process_query,
                        inputs=[query_input, top_k_slider],
                        outputs=[kb_answer_output, kb_sources_output, kb_debug_output]
                    )

                    # Show/hide debug info when checkbox changes
                    show_debug.change(
                        fn=lambda show: gr.update(visible=show),
                        inputs=[show_debug],
                        outputs=[kb_debug_output]
                    )

            # Footer with usage instructions
            gr.Markdown(
                """
                ---
                ### 📖 System Overview

                **🎯 Core Features:**
                1. **Bulk Classification Dashboard**: Process multiple tickets with automated classification
                2. **Interactive AI Agent**: Real-time ticket processing with classification and response
                3. **Ticket SQL DB**: View all database contents including tickets, classifications, answers, and human reviews
                4. **Knowledge Base Query**: Direct access to RAG system for documentation queries

                **� Technical Stack:**
                - **Classification**: Google Gemini for topic, sentiment, and priority analysis
                - **RAG Pipeline**: sentence-transformers + Neo4j + Gemini for automated responses
                - **Storage**: SQLite for tickets and Neo4j graph database for knowledge base
                - **Topics**: How-to, Product, Connector, Lineage, API/SDK, SSO, Glossary, Best practices, Sensitive data

                **📊 Routing Logic:**
                - **RAG Topics** (Auto-answered): How-to, Product, Best practices, API/SDK, SSO
                - **Human Topics** (Manual review): Connector, Lineage, Glossary, Sensitive data
                """,
                elem_classes=["container"]
            )

        self.interface = interface
        return interface

    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        if not self.interface:
            self.create_interface()

        default_kwargs = {
            'share': False,
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'show_error': True,
            'quiet': False
        }
        default_kwargs.update(kwargs)

        logger.info(f"Launching Gradio interface on http://{default_kwargs['server_name']}:{default_kwargs['server_port']}")
        if self.interface:
            return self.interface.launch(**default_kwargs)
        else:
            raise RuntimeError("Failed to create Gradio interface")
