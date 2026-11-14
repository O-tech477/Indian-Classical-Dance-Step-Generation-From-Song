-- Run this once in the Supabase dashboard: SQL Editor → New query → Run.
-- The Flask app talks to these tables with the service role key, which bypasses RLS.
-- Row Level Security stays on so the public anon key cannot read or write chats.

create table if not exists public.chats (
  id uuid primary key default gen_random_uuid(),
  title text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.chat_turns (
  id uuid primary key default gen_random_uuid(),
  chat_id uuid not null references public.chats (id) on delete cascade,
  prompt text not null,
  song_name text,
  composer text,
  raga text,
  tala text,
  tala_beats integer,
  youtube_url text,
  description text,
  filename text,
  audio_url text,
  audio_path text,
  dance_url text,
  created_at timestamptz not null default now()
);

create index if not exists chat_turns_chat_id_created_at_idx
  on public.chat_turns (chat_id, created_at);

create or replace function public.set_chat_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists chats_set_updated_at on public.chats;
create trigger chats_set_updated_at
  before update on public.chats
  for each row
  execute function public.set_chat_updated_at();

alter table public.chats enable row level security;
alter table public.chat_turns enable row level security;

-- One profile per Google account. The id is the Supabase auth user id.
create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  email text,
  full_name text,
  avatar_url text,
  created_at timestamptz not null default now()
);

alter table public.chats
  add column if not exists user_id uuid references auth.users (id) on delete cascade;

create index if not exists chats_user_id_updated_at_idx
  on public.chats (user_id, updated_at desc);

alter table public.profiles enable row level security;

drop policy if exists "read own profile" on public.profiles;
create policy "read own profile" on public.profiles
  for select using (auth.uid() = id);

drop policy if exists "own chats" on public.chats;
create policy "own chats" on public.chats
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "own chat turns" on public.chat_turns;
create policy "own chat turns" on public.chat_turns
  for all using (
    exists (
      select 1 from public.chats
      where chats.id = chat_turns.chat_id and chats.user_id = auth.uid()
    )
  ) with check (
    exists (
      select 1 from public.chats
      where chats.id = chat_turns.chat_id and chats.user_id = auth.uid()
    )
  );
