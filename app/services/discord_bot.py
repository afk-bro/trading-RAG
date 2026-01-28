"""Discord bot service using gateway connection.

Runs as a background task alongside FastAPI. Connects outbound to Discord
(no public URL needed). Slash commands work immediately after sync.

Usage:
    The bot starts automatically with the FastAPI app if DISCORD_BOT_TOKEN is set.
    Commands are synced to the guild specified by DISCORD_GUILD_ID (instant),
    or globally if not set (takes ~1 hour).
"""

import asyncio
from typing import Optional
from uuid import UUID

import discord
from discord import app_commands
import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

# Bot instance (created on startup)
_bot: Optional["TradingBot"] = None


class TradingBot(discord.Client):
    """Discord bot with slash commands for trading system."""

    def __init__(self, guild_id: Optional[int] = None):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.guild_id = guild_id
        self._setup_commands()

    def _setup_commands(self):
        """Register slash commands."""

        @self.tree.command(name="strategies", description="List trading strategies")
        @app_commands.describe(
            status="Filter by status (active, draft, paused, archived)",
            limit="Max results (default 10)",
        )
        async def strategies(
            interaction: discord.Interaction,
            status: Optional[str] = None,
            limit: int = 10,
        ):
            await self._cmd_strategies(interaction, status, limit)

        @self.tree.command(name="alerts", description="Show active operational alerts")
        @app_commands.describe(
            include_resolved="Include resolved alerts",
        )
        async def alerts(
            interaction: discord.Interaction,
            include_resolved: bool = False,
        ):
            await self._cmd_alerts(interaction, include_resolved)

        @self.tree.command(name="health", description="Check system health status")
        async def health(interaction: discord.Interaction):
            await self._cmd_health(interaction)

        @self.tree.command(name="pine", description="List Pine scripts")
        @app_commands.describe(
            search="Search by script name",
            limit="Max results (default 10)",
        )
        async def pine(
            interaction: discord.Interaction,
            search: Optional[str] = None,
            limit: int = 10,
        ):
            await self._cmd_pine(interaction, search, limit)

        @self.tree.command(name="help", description="Show available commands")
        async def help_cmd(interaction: discord.Interaction):
            await self._cmd_help(interaction)

    async def setup_hook(self):
        """Sync commands on startup."""
        if self.guild_id:
            guild = discord.Object(id=self.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            logger.info("discord_commands_synced", guild_id=self.guild_id)
        else:
            await self.tree.sync()
            logger.info("discord_commands_synced_globally")

    async def on_ready(self):
        """Log when bot is ready."""
        logger.info(
            "discord_bot_ready",
            user=str(self.user),
            guild_count=len(self.guilds),
        )

    # =========================================================================
    # Command Implementations
    # =========================================================================

    async def _cmd_strategies(
        self,
        interaction: discord.Interaction,
        status: Optional[str],
        limit: int,
    ):
        """Handle /strategies command."""
        pool = self._get_pool()
        if not pool:
            await interaction.response.send_message(
                ":x: Database not available", ephemeral=True
            )
            return

        query = """
            SELECT name, slug, engine, status,
                   (backtest_summary->>'best_oos_score')::float as oos_score
            FROM strategies
            WHERE ($1::text IS NULL OR status = $1)
            ORDER BY updated_at DESC
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, status, limit)

        if not rows:
            embed = discord.Embed(
                title="Strategies",
                description="No strategies found.",
                color=0xFEE75C,
            )
            await interaction.response.send_message(embed=embed)
            return

        lines = []
        for row in rows:
            status_emoji = {
                "active": ":green_circle:",
                "draft": ":white_circle:",
                "archived": ":black_circle:",
                "paused": ":yellow_circle:",
            }.get(row["status"], ":white_circle:")

            oos = f" (OOS: {row['oos_score']:.2f})" if row["oos_score"] else ""
            lines.append(f"{status_emoji} **{row['name']}** [{row['engine']}]{oos}")

        embed = discord.Embed(
            title=f"Strategies ({len(rows)})",
            description="\n".join(lines[:15]),
            color=0x57F287,
        )
        await interaction.response.send_message(embed=embed)

    async def _cmd_alerts(
        self,
        interaction: discord.Interaction,
        include_resolved: bool,
    ):
        """Handle /alerts command."""
        pool = self._get_pool()
        if not pool:
            await interaction.response.send_message(
                ":x: Database not available", ephemeral=True
            )
            return

        status_filter = "1=1" if include_resolved else "status = 'active'"

        query = f"""
            SELECT rule_type, severity, status, occurrence_count, acknowledged_at
            FROM ops_alerts
            WHERE {status_filter}
              AND created_at >= NOW() - INTERVAL '7 days'
            ORDER BY
                CASE severity
                    WHEN 'critical' THEN 0
                    WHEN 'high' THEN 1
                    WHEN 'medium' THEN 2
                    WHEN 'low' THEN 3
                END,
                created_at DESC
            LIMIT 15
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query)

        if not rows:
            embed = discord.Embed(
                title="Alerts",
                description=":white_check_mark: No active alerts!",
                color=0x57F287,
            )
            await interaction.response.send_message(embed=embed)
            return

        severity_emoji = {
            "critical": ":red_circle:",
            "high": ":orange_circle:",
            "medium": ":yellow_circle:",
            "low": ":blue_circle:",
        }

        lines = []
        for row in rows:
            emoji = severity_emoji.get(row["severity"], ":white_circle:")
            rule = row["rule_type"].replace("_", " ").title()
            count = (
                f" (x{row['occurrence_count']})" if row["occurrence_count"] > 1 else ""
            )
            ack = " :ballot_box_with_check:" if row["acknowledged_at"] else ""
            lines.append(f"{emoji} **{rule}**{count}{ack}")

        critical = sum(1 for r in rows if r["severity"] == "critical")
        high = sum(1 for r in rows if r["severity"] == "high")

        if critical > 0:
            title = f":rotating_light: {critical} CRITICAL"
            color = 0xED4245
        elif high > 0:
            title = f":warning: {high} High Priority"
            color = 0xFFA500
        else:
            title = "Active Alerts"
            color = 0xFEE75C

        embed = discord.Embed(
            title=f"{title} ({len(rows)})",
            description="\n".join(lines),
            color=color,
        )
        await interaction.response.send_message(embed=embed)

    async def _cmd_health(self, interaction: discord.Interaction):
        """Handle /health command."""
        import httpx

        settings = get_settings()
        embed = discord.Embed(title="System Health", color=0x5865F2)

        # Check database
        pool = self._get_pool()
        if pool:
            try:
                async with pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                embed.add_field(name="Database", value=":green_circle: OK", inline=True)
            except Exception as e:
                embed.add_field(
                    name="Database", value=f":red_circle: {str(e)[:20]}", inline=True
                )
        else:
            embed.add_field(
                name="Database", value=":red_circle: Not connected", inline=True
            )

        # Check Qdrant
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{settings.qdrant_url}/healthz")
                if resp.status_code == 200:
                    embed.add_field(
                        name="Qdrant", value=":green_circle: OK", inline=True
                    )
                else:
                    embed.add_field(
                        name="Qdrant",
                        value=f":yellow_circle: {resp.status_code}",
                        inline=True,
                    )
        except Exception:
            embed.add_field(
                name="Qdrant", value=":red_circle: Unreachable", inline=True
            )

        # Check Ollama
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{settings.ollama_base_url}/api/tags")
                if resp.status_code == 200:
                    embed.add_field(
                        name="Ollama", value=":green_circle: OK", inline=True
                    )
                else:
                    embed.add_field(
                        name="Ollama",
                        value=f":yellow_circle: {resp.status_code}",
                        inline=True,
                    )
        except Exception:
            embed.add_field(
                name="Ollama", value=":red_circle: Unreachable", inline=True
            )

        # Set color based on status
        field_values = [f.value for f in embed.fields]
        if any(":red_circle:" in v for v in field_values):
            embed.color = 0xED4245
            embed.title = ":x: System Degraded"
        elif any(":yellow_circle:" in v for v in field_values):
            embed.color = 0xFEE75C
            embed.title = ":warning: System Warning"
        else:
            embed.color = 0x57F287
            embed.title = ":white_check_mark: System Healthy"

        await interaction.response.send_message(embed=embed)

    async def _cmd_pine(
        self,
        interaction: discord.Interaction,
        search: Optional[str],
        limit: int,
    ):
        """Handle /pine command."""
        pool = self._get_pool()
        if not pool:
            await interaction.response.send_message(
                ":x: Database not available", ephemeral=True
            )
            return

        query = """
            SELECT title, status, metadata->>'script_type' as script_type
            FROM documents
            WHERE source_type = 'pine'
              AND ($1::text IS NULL OR title ILIKE '%' || $1 || '%')
            ORDER BY created_at DESC
            LIMIT $2
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, search, limit)

        if not rows:
            msg = "No Pine scripts found."
            if search:
                msg = f"No Pine scripts matching '{search}'."
            embed = discord.Embed(title="Pine Scripts", description=msg, color=0xFEE75C)
            await interaction.response.send_message(embed=embed)
            return

        lines = []
        for row in rows:
            script_type = row["script_type"] or "unknown"
            type_emoji = {
                "strategy": ":chart_with_upwards_trend:",
                "indicator": ":bar_chart:",
                "library": ":books:",
            }.get(script_type, ":page_facing_up:")
            status_emoji = (
                ":green_circle:" if row["status"] == "indexed" else ":white_circle:"
            )
            lines.append(f"{type_emoji} {status_emoji} **{row['title'][:40]}**")

        embed = discord.Embed(
            title=f"Pine Scripts ({len(rows)})",
            description="\n".join(lines[:15]),
            color=0x5865F2,
        )
        await interaction.response.send_message(embed=embed)

    async def _cmd_help(self, interaction: discord.Interaction):
        """Handle /help command."""
        embed = discord.Embed(
            title="Trading RAG Bot Commands",
            description="Available slash commands:",
            color=0x5865F2,
        )

        embed.add_field(
            name="/health",
            value="Check system health (database, Qdrant, Ollama)",
            inline=False,
        )
        embed.add_field(
            name="/strategies",
            value="List trading strategies\n`status`: filter by active/draft/paused/archived\n`limit`: max results (default 10)",
            inline=False,
        )
        embed.add_field(
            name="/alerts",
            value="Show operational alerts\n`include_resolved`: show resolved alerts too",
            inline=False,
        )
        embed.add_field(
            name="/pine",
            value="List Pine scripts in knowledge base\n`search`: filter by name\n`limit`: max results (default 10)",
            inline=False,
        )
        embed.add_field(
            name="/help",
            value="Show this help message",
            inline=False,
        )

        embed.set_footer(text="Trading RAG Pipeline")
        await interaction.response.send_message(embed=embed)

    def _get_pool(self):
        """Get database pool from ingest router."""
        try:
            from app.routers.ingest import _db_pool

            return _db_pool
        except Exception:
            return None


async def start_bot():
    """Start the Discord bot (call from FastAPI lifespan)."""
    global _bot

    settings = get_settings()
    token = settings.discord_bot_token

    if not token:
        logger.info("discord_bot_disabled", reason="no token configured")
        return

    # Get guild ID for instant command sync (optional)
    guild_id = getattr(settings, "discord_guild_id", None)
    if guild_id:
        guild_id = int(guild_id)

    _bot = TradingBot(guild_id=guild_id)

    # Run bot in background task
    asyncio.create_task(_run_bot(token))
    logger.info("discord_bot_starting")


async def _run_bot(token: str):
    """Run the bot (handles reconnection)."""
    global _bot
    try:
        await _bot.start(token)
    except Exception as e:
        logger.error("discord_bot_error", error=str(e))


async def stop_bot():
    """Stop the Discord bot (call from FastAPI shutdown)."""
    global _bot
    if _bot:
        await _bot.close()
        logger.info("discord_bot_stopped")
        _bot = None
